import json
import os
import random
import numpy as np
from datetime import datetime, timedelta, timezone

import torch

KST = timezone(timedelta(hours=9))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (may reduce performance)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def now_tag():
    return datetime.now(tz=KST).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
