# train.py

import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from data import VAL_RATE, get_dataloaders
from model import build_model
from utils import make_run_dir, save_json, set_seed, plot_history


def check_positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Train image classification model on MNIST")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mlp"],
                        help="Model type (default: cnn)")
    parser.add_argument("--epochs", type=check_positive, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=check_positive, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--save-best", action="store_true",
                        help="Save the best model based on validation accuracy")
    parser.add_argument("--patience", type=check_positive, default=5,
                        help="Early stopping patience based on val loss (default: 5)")
    parser.add_argument("--min-delta", type=float, default=0.0005,
                        help="Minimum val loss improvement to reset patience (default: 0.0005)")

    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    loss_sum, correct, n = 0.0, 0, 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        loss_sum += loss.item() * bs
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        n += bs

    avg_loss = loss_sum / n
    acc = correct / n
    return avg_loss, acc


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            bs = xb.size(0)
            loss_sum += loss.item() * bs
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            n += bs

    avg_loss = loss_sum / n
    acc = correct / n
    return avg_loss, acc


def main():
    args = parse_args()
    set_seed(args.seed)

    start_time = datetime.now().isoformat(timespec="seconds")
    start_perf = time.perf_counter()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = make_run_dir("outputs")
    print("Device:", device)
    print("Run dir:", run_dir)

    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size, args.seed)

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best = {"enabled": bool(args.save_best), "epoch": None, "val_acc": None, "path": None}
    best_path = os.path.join(run_dir, "best_model.pt")
    best_state = None

    # early stopping state
    best_val_es = None
    no_improve = 0
    early_stopped = False

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"\nEpoch {epoch}/{args.epochs}\n"
            f"Train loss = {train_loss:.4f}, Train acc = {train_acc:.4f}\n"
            f"Val loss   = {val_loss:.4f}, Val acc   = {val_acc:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if args.save_best and (best["val_acc"] is None or val_acc > best["val_acc"]):
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best["epoch"] = epoch
            best["val_acc"] = val_acc
            best["path"] = best_path

        if best_val_es is None or val_loss < best_val_es - args.min_delta:
            best_val_es = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"\nEarly stopping: no val loss improvement for {args.patience} epochs.")
            early_stopped = True
            break

    if best["enabled"] and best_state is not None:
        print(f"\nLoading best model from epoch {best['epoch']} with val acc {best['val_acc']:.4f}")
        torch.save(best_state, best_path)
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest loss = {test_loss:.4f}, Test acc = {test_acc:.4f}")

    metrics = {
        "meta": {
            "model": args.model,
            "epochs": args.epochs,
            "epochs_ran": len(history["train_loss"]),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "device": device,
            "val_rate": VAL_RATE,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "early_stopped": early_stopped,
            "start_time": start_time,
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "duration_sec": round(time.perf_counter() - start_perf, 3),
        },
        "metrics": {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "history": history,
            "best": best,
        },
    }

    save_json(os.path.join(run_dir, "metrics.json"), metrics)

    plot_history(run_dir)


if __name__ == "__main__":
    main()
