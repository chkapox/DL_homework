import csv
from pathlib import Path
from contextlib import nullcontext

import torch
import torchaudio
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm.auto import tqdm

from src.utils.io_utils import ROOT_PATH


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(cfg: DictConfig):
    device = (
        torch.device("cuda")
        if (cfg.trainer.device in ("auto", "cuda") and torch.cuda.is_available())
        else torch.device("cpu")
    )

    ds_cfg = cfg.datasets.test
    ds_cfg.split = cfg.datasets.test.split
    ds_cfg.shuffle_index = False
    dataset = instantiate(ds_cfg)

    collate = instantiate(cfg.dataloader.collate_fn)
    loader = instantiate(
        cfg.dataloader,
        dataset=dataset,
        shuffle=False,
        collate_fn=collate,
    )

    model = instantiate(cfg.model).to(device)
    ckpt_dir = ROOT_PATH / cfg.trainer.save_dir / cfg.writer.run_name
    ckpt_name = cfg.trainer.get("resume_from") or "model_best.pth"
    ckpt_path = ckpt_dir / ckpt_name
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state.get("state_dict", state))
    print(">>> checkpoint loaded from:", ckpt_path)
    print(">>> param abs-mean:", next(model.parameters()).abs().mean().item())
    model.eval()

    keys_in_order = [Path(e["path"]).stem for e in dataset._index]

    apply_tform = bool(getattr(cfg, "apply_tform", True))
    tform = None
    if apply_tform:
        try:
            tform_cfg = cfg.transforms.batch_transforms.inference.data_object
            tform = instantiate(tform_cfg)
            if isinstance(tform, torch.nn.Module):
                tform = tform.to(device)
        except Exception:
            tform = None

    score_type = str(getattr(cfg, "score_type", "logit_diff")).lower()
    assert score_type in {"p_bona", "p_spoof", "logit_diff", "neg_logit_diff"}

    out_csv = Path(getattr(cfg, "out_csv", "preds.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    seg_frames = int(getattr(cfg, "seg_frames", 600))
    hop_frames = int(getattr(cfg, "hop_frames", 300))

    total = 0
    idx = 0
    amp_ctx = (
        torch.amp.autocast(device_type="cuda")
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), out_csv.open("w", newline="") as f, amp_ctx:
        writer = csv.writer(f)
        for batch in tqdm(loader, total=len(loader), desc="inference"):
            x = batch["data_object"].to(device)
            if tform is not None:
                x = tform(x)

            if x.dim() == 3:
                x = x.unsqueeze(1)

            B, C, T, F = x.shape
            use_windows = (seg_frames > 0) and (seg_frames < T)

            if not use_windows:
                logits = model(data_object=x)["logits"]
            else:
                starts = list(range(0, max(1, T - seg_frames + 1), hop_frames))
                if starts[-1] != T - seg_frames:
                    starts.append(T - seg_frames)
                sum_logits = torch.zeros(B, 2, device=device, dtype=torch.float32)
                for s in starts:
                    x_win = x[:, :, s : s + seg_frames, :]
                    logits_win = model(data_object=x_win)["logits"]
                    sum_logits += logits_win
                logits = sum_logits / float(len(starts))

            probs = torch.softmax(logits, dim=-1)
            p_bona = probs[:, 1]
            p_spoof = probs[:, 0]
            ld = logits[:, 1] - logits[:, 0]

            if score_type == "p_bona":
                score = p_bona
            elif score_type == "p_spoof":
                score = p_spoof
            elif score_type == "logit_diff":
                score = ld
            else:
                score = -ld

            score = score.detach().cpu()
            bsz = score.shape[0]
            for j in range(bsz):
                writer.writerow([keys_in_order[idx + j], float(score[j])])
            idx += bsz
            total += bsz

    print(f"Saved CSV to: {out_csv} ({total} rows)")


if __name__ == "__main__":
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass
    main()
