import os
# To prevent windows multiprocessing memory problem before importing anomalib memory
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import json

from src.config import load_config
from src.data import get_datamodule
from src.model import build_model

from anomalib.engine import Engine
from lightning.pytorch import seed_everything

OUTPUT = Path("reports/figures")
CASES = ["TP", "FN", "FP"]



def collect_predicted_data(preds):
    rows, maps = [], {}

    for batch in preds:
        for i in range(len(batch.image_path)):
            path = batch.image_path[i]
            gt_label = int(batch.gt_label[i])
            pred_label = int(batch.pred_label[i])
            pred_score = float(batch.pred_score[i])

            rows.append({
                "image_path": path,
                "gt_label" : gt_label,
                "pred_label" : pred_label,
                "pred_score" : pred_score
            })

            maps[path] = (batch.anomaly_map[i].squeeze().cpu().numpy(),
                          batch.gt_mask[i].squeeze().cpu().numpy())

    df = pd.DataFrame(rows)
    df["case"] = np.select(
        [(df.gt_label == 1) & (df.pred_label == 1),
         (df.gt_label == 1) & (df.pred_label == 0),
         (df.gt_label == 0) & (df.pred_label == 1)],
        ["TP", "FN", "FP"], default="TN")
    return df, maps


def plot_case(df, maps, case, n=3):
    subset = df[df.case == case]
    # TP: most confident catches; FN/FP: the worst mistakes
    subset = subset.sort_values("pred_score", ascending=(case == "FN")).head(n)
    for _, row in subset.iterrows():
        amap, gmask = maps[row.image_path]
        img = Image.open(row.image_path).convert("RGB").resize(amap.shape[::-1])
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img); ax[0].set_title(f"{case} | score {row.pred_score:.3f}")
        ax[1].imshow(gmask, cmap="gray"); ax[1].set_title("Ground truth")
        ax[2].imshow(img); ax[2].imshow(amap, cmap="jet", alpha=0.5); ax[2].set_title("Anomaly map")
        for a in ax: a.axis("off")
        fig.savefig(OUTPUT / f"{case}_{Path(row.image_path).stem}.png", bbox_inches="tight")
        plt.close(fig)


def evaluate_model():
    # Get config
    cfg = load_config()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for old in OUTPUT.glob("*.png"): # Clear all previous saved images
        old.unlink()

    # Get data module
    seed_everything(42)
    data_module = get_datamodule(cfg["data"])

    # Build model
    model = build_model(cfg["model"])
    engine = Engine(default_root_dir = cfg["output_dir"])

    # Aggregate metrics
    metrics =  engine.test(model=model,
                datamodule= data_module,
                ckpt_path= cfg["export"]["ckpt"])
    with open("reports/metrics.json","w") as f:
        json.dump(metrics[0],f,indent=2)

    # Per-image predictions
    preds = engine.predict(model= model,
                dataloaders= data_module.test_dataloader(),
                ckpt_path= cfg["export"]["ckpt"])
    df,maps = collect_predicted_data(preds)

    # Save predicted result and case count
    df.to_csv("reports/per_image_predictions.csv", index=False)
    case_count = df.case.value_counts().to_dict()
    with open("reports/case_counts.json","w") as f:
        json.dump(case_count, f, indent=2)
    print(df.case.value_counts())  

    # Plot predicted result
    for case in CASES:
        plot_case(df, maps, case)



if __name__ == "__main__":
    evaluate_model()