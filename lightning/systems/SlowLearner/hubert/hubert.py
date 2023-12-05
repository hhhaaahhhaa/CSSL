import torch
import torch.nn as nn
import copy

from lightning.base.optimizer import get_optimizer
from lightning.base.scheduler import get_scheduler
from lightning.systems.CTrain.hubert import HubertSystem


class HubertSLSystem(HubertSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # separate optimizers
    def build_optimized_model(self) -> dict[str, nn.Module]:
        return {
            "slow": self.extractor,
            "fast": self.head,
        }

    def configure_optimizers(self):
        optimizers, schedulers = [], []
        optimized_modules = self.build_optimized_model()

        for name in ["slow", "fast"]:
            cfg = copy.deepcopy(self.train_config)
            cfg["optimizer"]["lr"] = self.algorithm_config["lr"][name]
            module = optimized_modules[name]
            cnt = sum([p.numel() for p in module.parameters() if p.requires_grad])
            print(f"Optimiable parameters ({name}): {cnt}")
            optimizer = get_optimizer(module, self.model_config, cfg)
            scheduler = {
                "scheduler": get_scheduler(optimizer, cfg),
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
        # correctly count global step
        # See issue at https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        opts = []
        for idx, opt in enumerate(self.optimizers()):
            if idx > 0:
                opt._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
                opt._on_after_step = lambda : self.trainer.profiler.stop("optimizer_step")
            opts.append(opt)
        
        # gradient accumulation
        self.manual_backward(loss / self.train_config["optimizer"]["grad_acc_step"])
        if (batch_idx + 1) % self.train_config["optimizer"]["grad_acc_step"] == 0:
            for opt in opts:
                # clip gradients
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.train_config["optimizer"]["grad_clip_thresh"],
                    gradient_clip_algorithm="norm"
                )
                opt.step()
            for opt in opts:
                opt.zero_grad()

    def training_step(self, batch, batch_idx):
        labels, _ = batch
        train_loss_dict, predictions, _ = self.common_step(batch, batch_idx, train=True)
        
        # print(self.global_step, batch_idx)
        self._custom_optimization(train_loss_dict["Total Loss"], batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions, '_batch': labels}
