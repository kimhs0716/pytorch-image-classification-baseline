# utils.py

import json
import os
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import matplotlib.pyplot as plt


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


def now_tag() -> str:
    """Get the current time in KST formatted as a string."""
    return datetime.now(KST).strftime('%y%m%d_%H%M%S')


def ensure_dir(dir_path: str) -> None:
    """Ensure that a directory exists; create it if it doesn't."""
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def make_run_dir(output_root: str) -> str:
    """Create a new run directory with a timestamp."""
    run_dir = os.path.join(output_root, now_tag())
    ensure_dir(run_dir)
    return run_dir


def plot_history(run_dir: str) -> None:
    """Generate plots for training history"""
    data_path = os.path.join(run_dir, "metrics.json")
    
    if not os.path.exists(data_path):
        print(f"Warning: metrics.json not found at {data_path}")
        return
    
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        history = data["metrics"]["history"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading metrics.json: {e}")
        return
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(history["train_loss"], label="Train Loss")
    ax[0].plot(history["val_loss"], label="Val Loss")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend(loc="upper right")

    ax[1].plot(history["train_acc"], label="Train Acc")
    ax[1].plot(history["val_acc"], label="Val Acc")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "history.png"))
    print(f"Saved training history plot to {os.path.join(run_dir, 'history.png')}")
    # plt.show()
