import torch
import torch.nn as nn

from lightning.base.optimizer import get_optimizer
from lightning.base.scheduler import get_scheduler
from lightning.systems.CTrain.hubert import HubertSystem


class HubertSLSystem(HubertSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # separate optimizers
    def build_optimized_model(self) -> list[nn.Module]:
        return [self.extractor, self.head]

    def configure_optimizers(self):
        # TODO: decompose train config into multiple configs here
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        optimizers, schedulers = [], []
        optimized_modules = self.build_optimized_model()
        for idx, optimized_module in optimized_modules:
            cnt = sum([p.numel() for p in optimized_module.parameters() if p.requires_grad])
            print(f"Optimiable parameters ({idx}): {cnt}")
            optimizer = get_optimizer(optimized_module, self.model_config, self.train_config)
            scheduler = {
                "scheduler": get_scheduler(optimizer, self.train_config),
                'interval': 'step', # "epoch" or "step"
                'frequency': 1,
                'monitor': self.default_monitor,
            }
            optimizers.append(optimizer)
            schedulers.append(scheduler)

        return optimizers, schedulers

    @property
    def automatic_optimization(self):
        return False
    
    def _custom_optimization(self, loss, batch_idx) -> None:
        # gradient accumulation
        self.manual_backward(loss / self.train_config["optimizer"]["grad_acc_step"])
        if (batch_idx + 1) % self.train_config["optimizer"]["grad_acc_step"] == 0:
            for opt in self.optimizers():
                # clip gradients
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.train_config["optimizer"]["grad_clip_thresh"],
                    gradient_clip_algorithm="norm"
                )
                opt.step()
                opt.zero_grad()

    def training_step(self, batch, batch_idx):
        labels, _ = batch
        train_loss_dict, predictions, _ = self.common_step(batch, batch_idx, train=True)
        
        self._custom_optimization(train_loss_dict["Total Loss"], batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions, '_batch': labels}
