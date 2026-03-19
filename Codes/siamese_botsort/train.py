# Usage: python train.py
import torch
from torch.utils.data import DataLoader
from dataset import SiameseDataset
from model import SiameseNet
import torch.nn as nn
from eval import compute_metrics
import random
import os
import csv
import time
from tqdm import tqdm

import matplotlib.pyplot as plt



class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, e1, e2, label):
        dist = torch.norm(e1 - e2, dim=1)
        loss = label * dist**2 + (1-label) * torch.clamp(self.margin - dist, min=0)**2
        return loss.mean()


def resolve_data_path(preferred_path, fallback_path=None):
    if os.path.isdir(preferred_path):
        return preferred_path
    if fallback_path and os.path.isdir(fallback_path):
        print(f"[Info] Using fallback dataset path: {fallback_path}")
        return fallback_path
    raise FileNotFoundError(
        f"Dataset folder not found. Checked: {preferred_path}"
        + (f" and {fallback_path}" if fallback_path else "")
    )


def print_metrics(title, metrics):
    print(
        f"{title} | "
        f"acc={metrics['accuracy']:.4f}, "
        f"prec={metrics['precision']:.4f}, "
        f"rec={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"tp={metrics['tp']}, tn={metrics['tn']}, fp={metrics['fp']}, fn={metrics['fn']}"
    )


def save_metrics_csv(path, rows):
    fieldnames = [
        "epoch",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
        "mean_same_distance",
        "mean_diff_distance",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _smooth(values, window=3):
    if len(values) < 2:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def plot_training_curves(history, out_path="training_metrics.png"):
    epochs = [row["epoch"] for row in history]

    metric_keys = [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mean_same_distance",
        "mean_diff_distance",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, key in enumerate(metric_keys):
        values = [row[key] for row in history]
        smooth_values = _smooth(values, window=3)

        axes[i].plot(epochs, values, marker="o", label="results", color="tab:blue")
        axes[i].plot(epochs, smooth_values, linestyle=":", linewidth=2, label="smooth", color="tab:orange")
        axes[i].set_title(key)
        axes[i].set_xlabel("epoch")
        axes[i].grid(alpha=0.25)

        if i == 0:
            axes[i].legend()

    # Last subplot: decision-quality trend from confusion matrix
    specificities = []
    for row in history:
        tn, fp = row["tn"], row["fp"]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)

    smooth_spec = _smooth(specificities, window=3)
    axes[7].plot(epochs, specificities, marker="o", label="results", color="tab:blue")
    axes[7].plot(epochs, smooth_spec, linestyle=":", linewidth=2, label="smooth", color="tab:orange")
    axes[7].set_title("specificity")
    axes[7].set_xlabel("epoch")
    axes[7].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved training curves to {out_path}")


def train_model(model, train_loader, eval_loader, criterion, optimizer, device, threshold=1.0, epochs=3):
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            ncols=100,
            leave=True,
        )

        for step, (img1, img2, label) in enumerate(progress_bar, start=1):
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.float().to(device)

            e1, e2 = model(img1, img2)
            loss = criterion(e1, e2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            running_avg = epoch_loss / step
            progress_bar.set_postfix(running_loss=f"{running_avg:.4f}")

        avg_loss = epoch_loss / len(train_loader)
        metrics = compute_metrics(model, eval_loader, device=device, threshold=threshold)
        elapsed = time.time() - t0

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Time: {elapsed:.1f}s", flush=True)
        print_metrics("Validation", metrics)

        history.append({
            "epoch": epoch + 1,
            "loss": round(avg_loss, 6),
            "accuracy": round(metrics["accuracy"], 6),
            "precision": round(metrics["precision"], 6),
            "recall": round(metrics["recall"], 6),
            "f1": round(metrics["f1"], 6),
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "mean_same_distance": round(metrics["mean_same_distance"], 6),
            "mean_diff_distance": round(metrics["mean_diff_distance"], 6),
        })

    return history


def plot_distances(model, dataloader, device, out_path="distance_histogram.png", show_plot=False):
    same_dist = []
    diff_dist = []

    model.eval()
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)

            e1, e2 = model(img1, img2)
            dist = torch.norm(e1 - e2, dim=1).cpu().numpy()

            for d, l in zip(dist, label):
                if int(l) == 1:
                    same_dist.append(float(d))
                else:
                    diff_dist.append(float(d))

    plt.figure(figsize=(8, 5))
    plt.hist(same_dist, bins=40, alpha=0.6, label="Same", color="tab:blue")
    plt.hist(diff_dist, bins=40, alpha=0.6, label="Different", color="tab:orange")
    plt.xlabel("Embedding distance")
    plt.ylabel("Count")
    plt.title("Distance Distribution (Same vs Different)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved histogram to {out_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()


def demo_random_pairs(model, dataset, device, n_pairs=3):
    model.eval()
    n_pairs = max(2, min(3, n_pairs))

    with torch.no_grad():
        for _ in range(n_pairs):
            idx = random.randint(0, len(dataset) - 1)
            img1, img2, label = dataset[idx]

            img1 = img1.unsqueeze(0).to(device)
            img2 = img2.unsqueeze(0).to(device)

            e1, e2 = model(img1, img2)
            dist = torch.norm(e1 - e2, dim=1).item()

            if int(label) == 1:
                print(f"Same object distance: {dist:.4f}")
            else:
                print(f"Different object distance: {dist:.4f}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def p(*parts):
        return os.path.join(script_dir, *parts)

    train_root = resolve_data_path(p("crops", "train"), p("crop", "train"))
    val_root = None
    if os.path.isdir(p("crops", "val")):
        val_root = p("crops", "val")
    elif os.path.isdir(p("crop", "val")):
        val_root = p("crop", "val")

    batch_size = 32
    epochs = 5
    threshold = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = SiameseDataset(train_root)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if val_root is not None:
        print(f"Using validation dataset: {val_root}")
        eval_dataset = SiameseDataset(val_root)
    else:
        print("[Info] Validation folder not found. Using train set for evaluation.")
        eval_dataset = train_dataset
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SiameseNet().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = train_model(
        model,
        train_loader,
        eval_loader,
        criterion,
        optimizer,
        device,
        threshold=threshold,
        epochs=epochs,
    )

    out_dir = p("eval_siamese")
    os.makedirs(out_dir, exist_ok=True)

    metrics_csv_path = os.path.join(out_dir, "metrics_log.csv")
    save_metrics_csv(metrics_csv_path, history)
    print(f"Saved metrics to {metrics_csv_path}")

    curves_path = os.path.join(out_dir, "training_metrics.png")
    plot_training_curves(history, out_path=curves_path)

    model_path = p("siamese_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    final_metrics = compute_metrics(model, eval_loader, device=device, threshold=threshold)
    print_metrics(f"Final (threshold={threshold})", final_metrics)

    plot_distances(model, eval_loader, device, out_path=os.path.join(out_dir, "distance_histogram.png"), show_plot=False)
    demo_random_pairs(model, eval_dataset, device, n_pairs=3)


if __name__ == "__main__":
    main()