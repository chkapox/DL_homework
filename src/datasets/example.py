import numpy as np
import torch
from tqdm.auto import tqdm
import os
from pathlib import Path
import torchaudio
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
class ExampleDataset(BaseDataset):

    def __init__(
        self,
        root: str,
        split: str,
            sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms=None,
        *args, **kwargs,
    ):

        self.root = Path(root) if Path(root).is_absolute() else (ROOT_PATH / root)
        split = split.lower()
        assert split in {"train", "dev", "eval"}, "split должен быть train/dev/eval"
        self.split = split

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}
        index = self._create_index()
        super().__init__(
            *args,
            **kwargs,
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def _protocol_path(self) -> Path:
        proto_dir = self.root / "ASVspoof2019_LA_cm_protocols"
        mapping = {
            "train": "ASVspoof2019.LA.cm.train.trn.txt",
            "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
            "eval":  "ASVspoof2019.LA.cm.eval.trl.txt",
        }
        return proto_dir / mapping[self.split]

    def _audio_dir(self) -> Path:
        split_dir = {"train": "ASVspoof2019_LA_train",
                     "dev":   "ASVspoof2019_LA_dev",
                     "eval":  "ASVspoof2019_LA_eval"}[self.split]
        return self.root / split_dir / "flac"

    def _parse_protocol_line(self, line: str):
        parts = line.strip().split()
        if not parts or line.startswith("#"):
            return None

        utt = next((t for t in parts if t.startswith(("LA_T_", "LA_D_", "LA_E_"))), None)
        if utt is None:
            return None

        label_str = parts[-1].lower()
        if label_str not in {"bonafide", "spoof"}:
            return None
        label = 1 if label_str == "bonafide" else 0
        return utt, label

    def _create_index(self) -> list[dict]:
        proto = self._protocol_path()
        audio_dir = self._audio_dir()
        assert proto.exists(), f"Не найден протокол: {proto}"
        assert audio_dir.exists(), f"Не найдена папка с аудио: {audio_dir}"

        index: list[dict] = []
        with open(proto, "r") as f:
            for line in f:
                parsed = self._parse_protocol_line(line)
                if parsed is None:
                    continue
                utt_id, label = parsed
                path_flac = audio_dir / f"{utt_id}.flac"
                path_wav  = audio_dir / f"{utt_id}.wav"
                if path_flac.exists():
                    path = path_flac
                elif path_wav.exists():
                    path = path_wav
                else:
                    continue
                index.append({"path": str(path), "label": label})
        assert len(index) > 0, f"Пустой индекс — проверь содержимое {audio_dir} и {proto}"
        return index

    def load_object(self, path: str) -> torch.Tensor:
        import torchaudio, torch
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wav = resampler(wav)

        stft = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=torch.blackman_window,
            power=2.0,
        )(wav)
        log_stft = torch.log1p(stft)
        return log_stft
