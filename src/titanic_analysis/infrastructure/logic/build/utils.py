import os
from pathlib import Path
import random

import joblib
import torch


def fix_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Python random
    random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms = True


def load_case_id(case_id_path: Path) -> int:
    if case_id_path.exists():
        case_id = joblib.load(case_id_path)
    else:
        case_id_path.parent.mkdir(exist_ok=True)
        case_id = 1

    return case_id