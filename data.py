# data.py

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DATA_DIR = "./data"
VAL_RATE = 0.1


def build_transform():
    """MNIST transform: PIL -> Tensor (0..1)."""
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def get_dataloaders(batch_size: int, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for MNIST."""
    tf = build_transform()

    train_full = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=tf)

    n_total = len(train_full)
    n_val = int(n_total * VAL_RATE) 
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
