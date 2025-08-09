import torch
from torch import nn
import torch.nn.functional as F


class Normalize1D(nn.Module):
    """
    Batch-version of Normalize for 1D Input.
    Used as an example of a batch transform.
    """

    def __init__(self, mean, std):
        """
        Args:
            mean (float): mean used in the normalization.
            std (float): std used in the normalization.
        """
        super().__init__()

        self.mean = mean
        self.std = std

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): normalized tensor.
        """
        x = (x - self.mean) / self.std
        return x

class Standardize2D(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

class CenterCropPadTime(nn.Module):
    def __init__(self, max_frames: int = 600):
        super().__init__()
        self.max_frames = max_frames

    def forward(self, x):
        B, C, Freq, T = x.shape
        if T > self.max_frames:
            start = (T - self.max_frames) // 2
            x = x[:, :, :, start:start + self.max_frames]
        elif T < self.max_frames:
            pad = self.max_frames - T
            x = F.pad(x, (0, pad, 0, 0))  # паддинг справа по времени
        return x