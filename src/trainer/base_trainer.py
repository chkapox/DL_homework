from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from torch import amp
from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from torch.utils.data import DataLoader
import gc
import os
import numpy as np

def _compute_eer_from_scores(probs: torch.Tensor, labels: torch.Tensor) -> float:
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


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        if isinstance(device, str):
            dev_str = device.lower()
            if dev_str == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(dev_str)
        else:
            self.device = device

        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        self._last_epoch = 0
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        self.save_period = self.cfg_trainer.save_period
        self.monitor = self.cfg_trainer.get("monitor", "off")

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]
            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.writer = writer

        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

        try:
            self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        except TypeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))

        self._best_thr = None

        self.config = config
        self.cfg_trainer = self.config.trainer

        if isinstance(device, str):
            dev_str = device.lower()
            if dev_str == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(dev_str)
        else:
            self.device = device

        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        self._last_epoch = 0
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        self.save_period = self.cfg_trainer.save_period
        self.monitor = self.cfg_trainer.get("monitor", "off")

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]
            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.writer = writer

        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

        try:
            self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        except TypeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))

        self._best_thr = None

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            logs = {"epoch": epoch}
            logs.update(result)

            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:
                break

    def _train_epoch(self, epoch):
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        if hasattr(self.criterion, "w"):
            w = getattr(self.criterion, "w")
            if isinstance(w, torch.Tensor) and w.numel() >= 2:
                w_cpu = w.detach().to("cpu").float()
                w0, w1 = float(w_cpu[0]), float(w_cpu[1])
                self.logger.info(f"[LOSS] epoch={epoch} class_weights w0={w0:.3f} w1={w1:.3f}")
                self.writer.add_scalar("LOSS_w0", w0)
                self.writer.add_scalar("LOSS_w1", w1)
                if not hasattr(self, "_w_snapshot"):
                    self._w_snapshot = (w0, w1)
                else:
                    if abs(w0 - self._w_snapshot[0]) > 1e-6 or abs(w1 - self._w_snapshot[1]) > 1e-6:
                        self.logger.warning(f"[LOSS] class_weights changed: was {self._w_snapshot}, now {(w0, w1)}")
                        self._w_snapshot = (w0, w1)

        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics
        self.writer.add_scalar("loss_train", logs["loss"])

        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        if os.name == "nt" and isinstance(dataloader, DataLoader) and dataloader.num_workers > 0:
            dataloader = DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
                collate_fn=getattr(dataloader, "collate_fn", None),
            )
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
                batch = self.process_batch(batch, metrics=self.evaluation_metrics)
                all_logits.append(batch["logits"].detach().cpu())
                all_labels.append(batch["labels"].detach().cpu())

            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(batch_idx, batch, part)

        if len(all_logits) == 0:
            logs = self.evaluation_metrics.result()
            logs.update({"Accuracy": 0.0, "F1": 0.0, "AUROC": 0.5, "EER": 0.5})
            self.writer.add_scalar(f"{part}_Accuracy", 0.0)
            self.writer.add_scalar(f"{part}_F1",       0.0)
            self.writer.add_scalar(f"{part}_AUROC",    0.5)
            self.writer.add_scalar(f"{part}_EER",      0.5)
            return logs

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0).long()
        p = torch.softmax(logits, dim=1)
        auroc0 = _roc_auc_torch(p[:, 0], labels)
        auroc1 = _roc_auc_torch(p[:, 1], labels)
        pos_idx = 0 if auroc0 > auroc1 else 1
        probs = p[:, pos_idx]
        self.writer.add_scalar(f"{part}_PosIndexUsed", float(pos_idx))
        try:
            bona_mean  = float(probs[labels == 1].mean()) if (labels == 1).any() else -1.0
            spoof_mean = float(probs[labels == 0].mean()) if (labels == 0).any() else -1.0
            self.logger.info(f"[{part.upper()}] pos_idx={pos_idx} auroc0={auroc0:.3f} auroc1={auroc1:.3f} bona_mean={bona_mean:.3f} spoof_mean={spoof_mean:.3f}")
        except Exception:
            pass

        def _best_thr_eer(scores: torch.Tensor, y: torch.Tensor, steps: int = 400) -> float:
            thrs = torch.linspace(0, 1, steps, device=scores.device)
            best_eer, best_t = 1.0, 0.5
            P = (y == 1).sum().item()
            N = (y == 0).sum().item()
            if P == 0 or N == 0:
                return 0.5
            for t in thrs:
                pr = (scores >= t).long()
                tp = ((pr==1)&(y==1)).sum().item()
                tn = ((pr==0)&(y==0)).sum().item()
                fp = ((pr==1)&(y==0)).sum().item()
                fn = ((pr==0)&(y==1)).sum().item()
                fpr = fp / max(fp+tn, 1)
                fnr = fn / max(fn+tp, 1)
                eer = 0.5 * (fpr + fnr)
                if eer < best_eer:
                    best_eer, best_t = eer, float(t)
            return best_t
        
        def _best_thr_f1(scores: torch.Tensor, y: torch.Tensor, steps: int = 400) -> float:
            thrs = torch.linspace(0, 1, steps, device=scores.device)
            best_f1, best_t = -1.0, 0.5
            for t in thrs:
                pr = (scores >= t).long()
                tp = ((pr==1)&(y==1)).sum().item()
                fp = ((pr==1)&(y==0)).sum().item()
                fn = ((pr==0)&(y==1)).sum().item()
                if tp == 0: 
                    continue
                prec = tp / (tp + fp + 1e-9)
                rec  = tp / (tp + fn + 1e-9)
                f1 = 2*prec*rec / (prec + rec + 1e-9)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            return best_t
        if part.lower().startswith("val"):
            best_thr = _best_thr_f1(probs, labels)
            self._best_thr = float(best_thr)
        elif part.lower().startswith("test") and (self._best_thr is not None):
            best_thr = torch.tensor(self._best_thr, device=probs.device)
        else:
            best_thr = _best_thr_f1(probs, labels)

        self.writer.add_scalar(f"{part}_BestThr", float(best_thr))

        preds = (probs >= best_thr).long()
        acc = (preds == labels).float().mean().item()
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        auroc = _roc_auc_torch(probs, labels)
        eer = _compute_eer_from_scores(probs, labels)

        logs = self.evaluation_metrics.result()
        logs["Accuracy"] = float(acc)
        logs["F1"]       = float(f1)
        logs["AUROC"]    = float(auroc)
        logs["EER"]      = float(eer)

        self.writer.add_scalar(f"{part}_Accuracy", acc)
        self.writer.add_scalar(f"{part}_F1",       f1)
        self.writer.add_scalar(f"{part}_AUROC",    auroc)
        self.writer.add_scalar(f"{part}_EER",      eer)
        with torch.no_grad():
            thr = float(best_thr)
            preds = (probs >= thr).long()
            pos_rate = float((preds == 1).float().mean())
            bona_mean = float(probs[labels==1].mean()) if (labels==1).any() else -1.0
            spoof_mean = float(probs[labels==0].mean()) if (labels==0).any() else -1.0
            tp = int(((preds==1)&(labels==1)).sum())
            fp = int(((preds==1)&(labels==0)).sum())
            fn = int(((preds==0)&(labels==1)).sum())
            tn = int(((preds==0)&(labels==0)).sum())
            self.logger.info(f"[{part.upper()}] pos@{thr:.2f}={pos_rate:.3f} cm=[tp={tp}, fp={fp}, fn={fn}, tn={tn}]")
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        self.writer.add_scalar(f"{part}_TPR", tpr)
        self.writer.add_scalar(f"{part}_FPR", fpr)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return logs

    def _monitor_performance(self, logs, not_improved_count):
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
