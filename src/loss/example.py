import torch
import torch.nn as nn
import torch.nn.functional as F

class ExampleLoss(nn.Module):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__()
        if class_weights is None:
            class_weights = [3.0, 1.0]
        w = torch.tensor(class_weights, dtype=torch.float)
        self.register_buffer("w", w)
        print(f"[ExampleLoss::__init__] class_weights from cfg = {class_weights}")

    def forward(self, *, logits=None, labels=None, **kwargs):
        assert logits is not None and labels is not None, "Pass logits and labels to loss"

        ce = F.cross_entropy(logits, labels.long(), reduction="none")

        if self.w is not None:
            w0, w1 = self.w[0].to(logits.device), self.w[1].to(logits.device)
        else:
            n_pos = (labels == 1).sum().clamp(min=1)
            n_neg = (labels == 0).sum().clamp(min=1)
            ratio = (n_neg.float() / n_pos.float()).detach()
            w0, w1 = 1.0, ratio

        sample_w = torch.where(labels.long() == 1, w1, w0).to(logits.device)

        loss = (ce * sample_w).mean()

        if not hasattr(self, "_printed"):
            try:
                print(f"[ExampleLoss] w0={float(w0):.3f} w1={float(w1):.3f}")
            except Exception:
                pass
            self._printed = True

        return {"loss": loss}