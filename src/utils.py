import random
from pathlib import Path

import numpy as np
import torch


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_device(device: str):
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def user_data_init(path: Path):
    for dir in ["runs", "state", "data"]:
        (path / dir).mkdir(parents=True, exist_ok=True)
