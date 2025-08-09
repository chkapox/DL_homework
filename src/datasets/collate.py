import torch
import torch.nn.functional as F

def collate_fn(dataset_items: list[dict]):
    # ожидаем data_object: (1, F, T)
    Fmax = 0
    Tmax = 0
    for it in dataset_items:
        _, Fcur, Tcur = it["data_object"].shape
        Fmax = max(Fmax, Fcur)
        Tmax = max(Tmax, Tcur)

    feats = []
    for it in dataset_items:
        x = it["data_object"]              # (1, F, T)
        pad_f = Fmax - x.shape[1]
        pad_t = Tmax - x.shape[2]
        if pad_f or pad_t:
            x = F.pad(x, (0, pad_t, 0, pad_f))  # справа по T, снизу по F
        feats.append(x)

    return {
        "data_object": torch.stack(feats, dim=0),  # (B, 1, Fmax, Tmax)
        "labels": torch.tensor([it["labels"] for it in dataset_items]),
    }
