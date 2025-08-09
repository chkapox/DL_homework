import torch
from src.trainer.base_trainer import BaseTrainer
from src.metrics.tracker import MetricTracker

class Trainer(BaseTrainer):
    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        use_amp = (
            getattr(self, "scaler", None) is not None
            and self.device.type == "cuda"
        )

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(
                logits=batch["logits"], labels=batch["labels"], features=batch.get("features")
            )

            batch.update(all_losses)

        if self.is_train:
            if use_amp:
                self.scaler.scale(batch["loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                self._clip_grad_norm()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                batch["loss"].backward()
                self._clip_grad_norm()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch