from typing import Any, Optional
import torch
import torch.nn as nn
import copy
from tqdm import tqdm

from lightning.base.optimizer import get_optimizer
from lightning.base.scheduler import get_scheduler
from lightning.systems.plugin import ITaskBoundary
from lightning.systems.CTrain.hubert import HubertSystem
from lightning.systems.SPU.hubert.datamodule import DataModule


class HubertSLSystem(HubertSystem, ITaskBoundary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # separate optimizers
    def build_optimized_model(self) -> dict[str, nn.Module]:
        return {
            "slow": self.extractor,
            "fast": self.head,
            "warmup": self.head,
        }

    def configure_optimizers(self):
        self.optimizer_keys = []  # pytorch lightning can only handle list of optimizers (no dict), so we record keys here
        optimizers, schedulers = [], []
        optimized_modules = self.build_optimized_model()

        for name in optimized_modules:
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
            self.optimizer_keys.append(name)

        return optimizers, schedulers

    @property
    def automatic_optimization(self):
        return False
    
    # Task boundary interface
    def is_task_boundary(self) -> bool:
        task_boundaries = self.task_config.get_info()["training"]["task_boundaries"]
        if not task_boundaries:
            return False
        return (self.global_step in task_boundaries)
    
    def task_init(self) -> None:
        # create task dataloader
        dm = self.trainer.datamodule  # link to datamodule
        assert isinstance(dm, DataModule)
        info = dm.task_config.get_info()
        tid = info["tid_seq"][self.global_step]
        print("Task start", tid)
        task_dataloader = dm.task_dataloader(tid, batch_size=self.bs)

        self._warmup(task_dataloader)
    
    def task_end(self) -> None:
        pass

    # Task warmup
    def _warmup(self, task_dataloader) -> None:
        task_warmup_step = self.algorithm_config.get("task_warmup_step", -1)
        if task_warmup_step < 0:
            return
        
        print("Task warmup...")
        cnt = 0
        total_step = task_warmup_step * self.train_config["optimizer"]["grad_acc_step"]
        while cnt < total_step:
            for batch_idx, batch in tqdm(enumerate(task_dataloader), total=total_step):
                train_loss_dict, _, _ = self.common_step(batch, batch_idx, train=True)
                self._warmup_optimization(train_loss_dict["Total Loss"], batch_idx)
                cnt += 1
                if cnt == total_step:
                    break
        # zero grad
        self.zero_grad(set_to_none=True)
        print("Done.")

    def _warmup_optimization(self, loss, batch_idx) -> None:
        opt = self.optimizers()[self.optimizer_keys.index("warmup")]

        # gradient accumulation
        self.manual_backward(loss / self.train_config["optimizer"]["grad_acc_step"])
        if (batch_idx + 1) % self.train_config["optimizer"]["grad_acc_step"] == 0:
            # clip gradients
            self.clip_gradients(
                opt,
                gradient_clip_val=self.train_config["optimizer"]["grad_clip_thresh"],
                gradient_clip_algorithm="norm"
            )
            opt.step()
            opt.zero_grad()

    # training
    def on_train_start(self) -> None:
        # prevent duplicate init or end since gradient accumulation return same global step multiple times
        self._is_task_init = False 
    
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if self.is_task_boundary():
            if not self._is_task_init:
                if self.global_step > 0:  # skip the first boundary which is 0
                    self.task_end()
                self.task_init()
                self._is_task_init = True
        else:
            self._is_task_init = False
    
    def training_step(self, batch, batch_idx):
        labels, _ = batch
        train_loss_dict, predictions, _ = self.common_step(batch, batch_idx, train=True)
        
        # print(self.global_step, batch_idx)
        self._custom_optimization(train_loss_dict["Total Loss"], batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions, '_batch': labels}
    
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
