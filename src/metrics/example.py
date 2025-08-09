import torch
from src.metrics.base_metric import BaseMetric

def _roc_auc_torch(probs: torch.Tensor, labels: torch.Tensor) -> float:
    order = torch.argsort(probs, descending=True)
    y = labels[order].int()
    P = (y == 1).sum().item()
    N = (y == 0).sum().item()
    if P == 0 or N == 0:
        return 0.5
    tp = torch.cumsum((y == 1).float(), dim=0)
    fp = torch.cumsum((y == 0).float(), dim=0)
    tpr = tp / P
    fpr = fp / N
    device = probs.device
    tpr = torch.cat([torch.tensor([0.0], device=device), tpr, torch.tensor([1.0], device=device)])
    fpr = torch.cat([torch.tensor([0.0], device=device), fpr, torch.tensor([1.0], device=device)])
    return torch.trapz(tpr, fpr).item()

def _eer_from_scores(probs: torch.Tensor, labels: torch.Tensor) -> float:
    order = torch.argsort(probs, descending=True)
    y = labels[order].int()
    P = (y == 1).sum().item()
    N = (y == 0).sum().item()
    if P == 0 or N == 0:
        return 0.5
    tp = torch.cumsum((y == 1).float(), dim=0)
    fp = torch.cumsum((y == 0).float(), dim=0)
    tpr = tp / P
    fpr = fp / N
    fnr = 1 - tpr
    i = torch.argmin(torch.abs(fpr - fnr))
    return 0.5 * (float(fpr[i]) + float(fnr[i]))

class ExampleMetric(BaseMetric):
    def __init__(self, metric=None, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._name = (kwargs.get("name", "Metric") or "Metric").lower()
        self.pos_index = kwargs.get("pos_index", 1)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        labels = labels.long()
        probs  = torch.softmax(logits, dim=1)[:, self.pos_index]

        name = self._name
        if "eer" in name:
            return _eer_from_scores(probs, labels)
        elif "auroc" in name:
            return _roc_auc_torch(probs, labels)
        elif "accuracy" in name:
            preds = (probs >= 0.5).long()
            return (preds == labels).float().mean().item()
        elif "f1" in name:
            preds = (probs >= 0.5).long()
            tp = ((preds == 1) & (labels == 1)).sum().item()
            fp = ((preds == 1) & (labels == 0)).sum().item()
            fn = ((preds == 0) & (labels == 1)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        else:
            preds = (probs >= 0.5).long()
            return (preds == labels).float().mean().item()
