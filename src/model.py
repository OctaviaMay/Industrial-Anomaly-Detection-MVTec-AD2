import os
# To prevent windows multiprocessing memory problem before importing anomalib memory
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from src.config import load_config

from anomalib.models import Patchcore
from anomalib.metrics import Evaluator, AUROC, F1Score, AUPRO


def build_model(model_cfg):
    # Build custom evaluator
    evaluator = Evaluator(test_metrics=[
                AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
                F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
                AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
                AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            ])

    return Patchcore(backbone= model_cfg['backbone'],
              layers=model_cfg['layers'],
              coreset_sampling_ratio= model_cfg['coreset_sampling_ratio'],
              evaluator=evaluator)


if __name__ == "__main__":
    cfg = load_config()
    build_model(cfg["model"])