from typing import Any, Optional
import torch
import torch.nn as nn
import copy
from tqdm import tqdm

from lightning.utils.tool import get_module_by_name
from lightning.base.optimizer import get_optimizer
from lightning.base.scheduler import get_scheduler
from lightning.systems.plugin import ITaskBoundary
from lightning.systems.CTrain import hubert
from lightning.systems.CTrain.hubert.model import HubertCustom
from lightning.systems.SPU.hubert.datamodule import DataModule


class HubertLoRAEWCSystem(hubert.HubertSystem, ITaskBoundary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._importances = None
        self._history = None

    # separate optimizers
    def build_optimized_model(self) -> dict[str, nn.Module]:
        return {
            "backbone": nn.ModuleList(self._localize()),
            "head": self.head,
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
        # create task dataloader
        dm = self.trainer.datamodule  # link to datamodule
        assert isinstance(dm, DataModule)
        info = dm.task_config.get_info()
        tid = info["tid_seq"][self.global_step - 1]
        print("Task end", tid)
        task_dataloader = dm.task_dataloader(tid, batch_size=self.bs, infinite=False)
        self._calc_fisher_matrix(task_dataloader)
        self._history = self._snapshot()
   
    # Task warmup
    def _warmup(self, task_dataloader) -> None:
        task_warmup_step = self.algorithm_config.get("task_warmup_step", -1)
        if task_warmup_step < 0:
            return
        
        print("Task warmup...")
        cnt = 0
        total_step = task_warmup_step * self.train_config["optimizer"]["grad_acc_step"]
        while cnt < total_step:
            for batch_idx, batch in tqdm(enumerate(task_dataloader)):
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

    # EWC
    def _calc_fisher_matrix(self, task_dataloader):
        print("Calculate fisher matrix...")

        # compute statisitcs from task dataloader
        self.eval()
        self._importances = []
        modules = self._localize()
        with torch.no_grad():
            for module in modules:
                self._importances.append({
                    name: torch.zeros_like(p.data) for name, p in module.named_parameters()
                })

        length = self.algorithm_config.get("n_estimate_batch", len(task_dataloader))
        for batch_idx, batch in tqdm(enumerate(task_dataloader), total=length):
            train_loss_dict, _, _ = self.common_step(batch, batch_idx, train=True)
            loss = train_loss_dict["Total Loss"]
            self.manual_backward(loss)

            with torch.no_grad():
                for importances, module in zip(self._importances, modules):
                    for name, p in module.named_parameters():
                        importances[name] += p.grad.data.clone().pow(2)
            self.zero_grad()
            if batch_idx + 1 == length:
                break

        # average over dataset length
        for importances in self._importances:
            for _, imp in importances.items():
                imp /= float(length)

        # zero grad
        self.zero_grad(set_to_none=True)
        self.train()
        print("Done.")
    
    def _snapshot(self):
        return None # LoRA_EWC avoids copying model
    
    def _ewc_loss(self):
        if self._importances is None:
            return torch.zeros(1, device=self.device)
        
        ewc_loss = 0
        modules = self._localize()
        for importances, module in zip(self._importances, modules):
            for name, p in module.named_parameters():
                ewc_loss += torch.sum(importances[name] * (p ** 2))
        return ewc_loss
    
    def _localize(self) -> list[nn.Module]:
        """ Get specific modules, here we use first layer of MLP in transformer encoder """
        res = []
        assert isinstance(self.extractor._model, HubertCustom)
        for i in range(12):
            module = get_module_by_name(self.extractor._model, f"model.encoder.layers.{i}.fc1")  # nn.Linear
            res.append(module)
        return res
    
    # training
    def on_train_start(self) -> None:
        # prevent duplicate init or end since gradient accumulation return same global step multiple times
        self._is_task_init = False 
        self._is_task_end = False
    
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if self.is_task_boundary():
            if not self._is_task_init:
                self.task_init()
                self._is_task_init = True
                self._is_task_end = False

    def on_train_batch_end(self, output: Any, batch: Any, batch_idx: int) -> None:
        if self.is_task_boundary() and self.global_step > 0:  # skip the first boundary which is 0
            if not self._is_task_end:
                self.task_end()
                self._is_task_init = False
                self._is_task_end = True
    
    def training_step(self, batch, batch_idx):
        labels, _ = batch
        train_loss_dict, predictions, _ = self.common_step(batch, batch_idx, train=True)
        ewc_loss = self.algorithm_config["ewc_loss_weight"] * self._ewc_loss()
        train_loss_dict = {
            "Total Loss": train_loss_dict["Total Loss"] + ewc_loss,
            "EWC Loss": ewc_loss
        }
        
        # print(self.global_step, batch_idx)
        self._custom_optimization(train_loss_dict["Total Loss"], batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions, '_batch': labels}

    def _custom_optimization(self, loss, batch_idx) -> None:
        # remove warmup optimizer
        opts = {self.optimizer_keys[idx]: x for idx, x in enumerate(self.optimizers()) if self.optimizer_keys[idx] != "warmup"}

        # correctly count global step
        # See issue at https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        for idx, key in enumerate(opts):
            if idx > 0:
                opts[key]._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
                opts[key]._on_after_step = lambda : self.trainer.profiler.stop("optimizer_step")
        
        # gradient accumulation
        self.manual_backward(loss / self.train_config["optimizer"]["grad_acc_step"])
        if (batch_idx + 1) % self.train_config["optimizer"]["grad_acc_step"] == 0:
            for key in opts:
                # clip gradients
                self.clip_gradients(
                    opts[key],
                    gradient_clip_val=self.train_config["optimizer"]["grad_clip_thresh"],
                    gradient_clip_algorithm="norm"
                )
                opts[key].step()
            for key in opts:
                opts[key].zero_grad()
