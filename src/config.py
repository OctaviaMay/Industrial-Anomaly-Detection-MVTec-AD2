from pathlib import Path
import yaml


def load_config(config_path: str = "configs/patchcore.yaml") ->dict:
    with open(Path(config_path), "r", encoding = "utf-8") as f:
        return yaml.safe_load(f) 