from anomalib.models import Patchcore
from anomalib.engine import Engine
from pathlib import Path
import json

from src.config import load_config
from src.data import get_datamodule

def train_model():
    # Get config
    cfg = load_config()

    # Initialize engine
    engine = Engine(default_root_dir=cfg["output_dir"])

    # Initialize model
    m = cfg["model"]
    model = Patchcore(
        backbone=m["backbone"],
        layers=m["layers"],
        coreset_sampling_ratio=m["coreset_sampling_ratio"],
        pre_trained=True,
    )
    
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
