import os
# To prevent windows multiprocessing memory problem before importing anomalib memory
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import time
import json
import torch
import onnxruntime as ort

from pathlib import Path
from PIL import Image

from src.config import load_config
from src.model import build_model

N_THREADS = 4 # Set same thread count to get fair comparison
ONNX_PATH = "results/export/weights/onnx/model.onnx" # exported ONNX model path
N_IMAGES = 40 # 20 for bad images, 20 for good images
EXP_BATCH = 2 # fixed by exported ONNX model input shape
WARMUP = 3


def load_images(img_root_path, img_size):
    paths = sorted((Path(img_root_path) / "test_public/good").glob("*.png"))[: N_IMAGES // 2] + \
            sorted((Path(img_root_path) / "test_public/bad").glob("*.png"))[: N_IMAGES // 2]
    
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(img_size) # Making sure ONNX model requires 3 color channel
        arr = np.asarray(img, dtype=np.float32)   # picture -> numbers, shape (H, W, 3), values 0-255
        arr = arr.transpose(2, 0, 1)              # channels first, shape (3, H, W)
        arr = arr / 255.0                         # scale values to 0-1
        imgs.append(arr)
        
    return np.stack(imgs)     
        

def run_torch(model, batch):
    with torch.no_grad():
        return model(torch.from_numpy(batch))

def run_onnx(sess, batch):
    return sess.run(["pred_score", "anomaly_map"], {"input": batch})


def time_runs(fn, batches):
    for b in batches[:WARMUP]:
        fn(b)
    start = time.perf_counter()
    for b in batches:
        fn(b)
    return (time.perf_counter() - start) * 1000 / (len(batches) * EXP_BATCH)


def benchmark():
    # 1. Get config
    cfg = load_config()
    img_size = tuple(cfg["export"]["input_size"])
    print("1. config loaded")

    # 2. Build PyTorch model
    torch.set_num_threads(N_THREADS)
    model = build_model(cfg["model"])
    ckpt = torch.load(f= cfg["export"]["ckpt"],
                      map_location="cpu",
                      weights_only=False)
    model.load_state_dict(state_dict=ckpt["state_dict"])
    model.eval()
    print("2. torch model ready")

    # 3. Prepare ONNX runtime session
    ort_sessopts = ort.SessionOptions()
    ort_sessopts.intra_op_num_threads = N_THREADS
    ort_sess = ort.InferenceSession(path_or_bytes= ONNX_PATH,
                         sess_options= ort_sessopts,
                         providers=["CPUExecutionProvider"])
    print("3. onnx session ready")

    # 4. Load images
    images = load_images(cfg["data"]["root"], img_size)
    batches = [images[i: i + EXP_BATCH] for i in range(0,len(images), EXP_BATCH)]
    print("4. images loaded:", images.shape)

    # 5. Verification: same input -> same output
    score_diffs, map_diffs = [], []
    for b in batches:
        t = run_torch(model, b)
        o_score, o_map = run_onnx(ort_sess,b)
        score_diffs.append(np.abs(t.pred_score.numpy() - o_score).max())
        map_diffs.append(np.abs(t.anomaly_map.numpy() - o_map).max())

    # 6. Latency
    torch_ms = time_runs(lambda b: run_torch(model, b), batches)
    onnx_ms = time_runs(lambda b: run_onnx(ort_sess, b), batches)

    # 7. Construct the result benchmark json
    result = {
        "images": len(images),
        "threads": N_THREADS,
        "pytorch_ms_per_image": round(torch_ms, 1),
        "onnx_ms_per_image": round(onnx_ms, 1),
        "speedup": round(torch_ms / onnx_ms, 2),
        "max_score_diff": float(max(score_diffs)),
        "max_map_diff": float(max(map_diffs)),
        "scores_match": bool(max(score_diffs) < 1e-3),
        "onnx_size_mb": round(Path(ONNX_PATH).stat().st_size / 1e6, 1)
    }

    print(json.dumps(result, indent=2))
    Path("reports").mkdir(exist_ok=True)
    with open("reports/benchmark.json", "w") as f:
        json.dump(result,f,indent=2)




if __name__ == "__main__":
    benchmark()