
import os
# To prevent windows multiprocessing memory problem before importing anomalib memory
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from src.config import load_config
from src.data import get_datamodule
from src.model import build_model

from anomalib.engine import Engine
from pathlib import Path
from lightning.pytorch import seed_everything
import json



def train_model():
    # Get config
    cfg = load_config()

    # Initialize engine
    engine = Engine(default_root_dir=cfg["output_dir"])

    # Initialize model
    m = cfg["model"]
    model = build_model(m)

    seed_everything(42)
    data_module = get_datamodule(cfg["data"])

    # Train model using the engine
    engine.fit(model=model, datamodule=data_module)
    results = engine.test(model=model, datamodule=data_module)
    print(results)

    # Save result
    Path("reports").mkdir(exist_ok=True)
    with open("reports/baseline_metrics.json", "w") as f:
        json.dump(results[0], f, indent=2)


if __name__ == "__main__":
    train_model()
