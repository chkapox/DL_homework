import torch
import torch.nn as nn



class MFM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        c = x.size(1)
        assert c % 2 == 0, "MFM expects even number of channels"
        a, b = x.split(c // 2, dim=1)
        return torch.max(a, b)
    
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        s = self.pool(x)
        s = self.fc(s)
        return x * s
    
class ConvMFM(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False, use_se=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size=k, stride=s, padding=p, bias=bias)
        self.mfm = MFM()
        self.se = SE(out_ch) if use_se else None
    def forward(self, x):
        y = self.mfm(self.conv(x))
        if self.se is not None:
            y = self.se(y)
        return y


class LCNN(nn.Module):
    def __init__(self, in_channels=1, dropout=0.5, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvMFM(in_channels, 64, k=5, s=1, p=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvMFM(64, 96, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvMFM(96, 128, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvMFM(128, 128, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(128, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")

    def forward(self, data_object, **batch):
        x = data_object
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.mean(dim=[2, 3])
        x = self.dropout(x)
        logits = self.classifier(x)
        return {"logits": logits, "features": x}

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
