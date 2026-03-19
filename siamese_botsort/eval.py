# Usage: import only — not run directly. Used by train.py via compute_metrics.
import torch

def compute_metrics(model, dataloader, device, threshold=1.0):
    model.eval()

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = 0

    same_distances = []
    diff_distances = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.float().to(device)

            e1, e2 = model(img1, img2)
            dist = torch.norm(e1 - e2, dim=1)
            preds = (dist < threshold).float()

            total += label.size(0)

            tp += ((preds == 1) & (label == 1)).sum().item()
            tn += ((preds == 0) & (label == 0)).sum().item()
            fp += ((preds == 1) & (label == 0)).sum().item()
            fn += ((preds == 0) & (label == 1)).sum().item()

            dist_cpu = dist.detach().cpu().numpy()
            label_cpu = label.detach().cpu().numpy()
            for d, l in zip(dist_cpu, label_cpu):
                if int(l) == 1:
                    same_distances.append(float(d))
                else:
                    diff_distances.append(float(d))

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    mean_same_dist = sum(same_distances) / len(same_distances) if same_distances else 0.0
    mean_diff_dist = sum(diff_distances) / len(diff_distances) if diff_distances else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "mean_same_distance": mean_same_dist,
        "mean_diff_distance": mean_diff_dist,
    }


def compute_accuracy(model, dataloader, device, threshold=1.0):
    return compute_metrics(model, dataloader, device, threshold=threshold)["accuracy"]