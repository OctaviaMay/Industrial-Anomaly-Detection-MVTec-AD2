import os
# To prevent windows multiprocessing memory problem before importing anomalib memory
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from pathlib import Path
from src.config import load_config
from src.model import build_model

from anomalib.engine import Engine
from anomalib.deploy import ExportType


def export_onnx():
    cfg = load_config()
    model = build_model(cfg["model"])

    ckpt = torch.load(f = cfg["export"]["ckpt"],
                      map_location= "cpu",
                      weights_only= False)
    model.load_state_dict(state_dict= ckpt["state_dict"])
    
    engine = Engine(default_root_dir=cfg["output_dir"])
    export_path = engine.export(
        model= model,
        export_type= ExportType.ONNX,
        export_root= Path(cfg["export"]["root"]),
        input_size= tuple(cfg["export"]["input_size"]),
        onnx_kwargs={"opset_version":18}
    )

    print(f"Exported to : {export_path}")


if __name__ == "__main__":
    export_onnx()
