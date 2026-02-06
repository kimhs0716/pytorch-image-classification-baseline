import json
import os
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import torch

KST = timezone(timedelta(hours=9))


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_json(filepath: str, data: dict) -> None:
    """Save data as a JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate accuracy from model logits and true labels."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def now_tag() -> str:
    """Get the current time in KST formatted as a string."""
    return datetime.now(KST).strftime('%Y%m%d_%H%M%S')


def ensure_dir(dir_path: str) -> None:
    """Ensure that a directory exists; create it if it doesn't."""
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def make_run_dir(output_root: str) -> str:
    """Create a new run directory with a timestamp."""
    run_dir = os.path.join(output_root, now_tag())
    ensure_dir(run_dir)
    return run_dir