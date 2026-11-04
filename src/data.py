
import os
# To prevent windows multiprocessing memory problem before importing anomalib memory
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


from anomalib.data import Folder
from src.config import load_config


def get_datamodule(data_cfg) -> Folder:
    datamodule = Folder(
        name='vial',
        root= data_cfg["root"],
        normal_dir = data_cfg["normal_dir"],
        normal_test_dir = data_cfg["normal_test_dir"],
        abnormal_dir = data_cfg["abnormal_dir"],
        mask_dir = data_cfg["mask_dir"],
        train_batch_size = data_cfg["batch_size"],
        eval_batch_size = data_cfg["batch_size"],
        num_workers = data_cfg["num_workers"]
    )

    datamodule.setup()
    return datamodule

   



if __name__ == "__main__":
    cfg = load_config()
    data_module = get_datamodule(cfg["data"])
    batch = next(iter(data_module.train_dataloader()))
    print(type(batch))
    print(batch.image.shape)

    print(f"train dataset: {len(data_module.train_data)}")
    print(f"test dataset: {len(data_module.test_data)}")