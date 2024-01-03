from typing import Any, Optional
import torch
import torch.nn as nn
import copy
from tqdm import tqdm

from lightning.utils.tool import get_module_by_name
from lightning.base.optimizer import get_optimizer
from lightning.base.scheduler import get_scheduler
from lightning.systems.CTrain import hubert
from lightning.systems.CTrain.hubert.model import HubertCustom
from .datamodule import DataModule


class HubertSPUSystem(hubert.HubertSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # separate optimizers
    def build_optimized_model(self) -> dict[str, nn.Module]:
        return {
            "backbone": self.extractor,
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

    # Task init
    def is_task_boundary(self):
        task_boundaries = self.task_config.get_info()["training"]["task_boundaries"]
        if not task_boundaries:
            return False
        try:
            res = (task_boundaries[self.next_boundary_idx] == self.global_step)
        except:
            self.next_boundary_idx = 0
            res = (task_boundaries[self.next_boundary_idx] == self.global_step)
        if res:
            self.next_boundary_idx += 1
        return res
    
    def task_init(self) -> None:
        # create task dataloader
        dm = self.trainer.datamodule  # link to datamodule
        assert isinstance(dm, DataModule)
        info = dm.task_config.get_info()
        tid = info["tid_seq"][self.global_step]
        print("Task ", tid)
        task_dataloader = dm.task_dataloader(tid, batch_size=self.bs)

        self._warmup(task_dataloader)
        self._select_parameter(task_dataloader)
    
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if self.is_task_boundary():
            self.task_init()
    
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

    # Selective Parameter Update (SPU)    
    def _select_parameter(self, task_dataloader) -> None:
        print("Selecting parameters...")
        # compute statisitcs from task dataloader
        cnt = 0
        n_estimate = self.algorithm_config["n_estimate"]
        for batch_idx, batch in enumerate(task_dataloader):
            train_loss_dict, _, _ = self.common_step(batch, batch_idx, train=True)
            loss = train_loss_dict["Total Loss"]
            self.manual_backward(loss)
            cnt += self.bs
            if cnt >= n_estimate:
                break
        
        # select and create gradient masks
        with torch.no_grad():
            self._spu_mask = []
            modules = self._localize()
            for module in modules:
                grad = module.weight.grad.data
                mask = torch.zeros_like(grad)
                h, w = mask.shape
                n_selected = int(h * w * self.algorithm_config["selection_rate"])
                _, idxs_flatten = torch.topk(torch.abs(grad.flatten()), n_selected)
                mask[(idxs_flatten // w).long(), (idxs_flatten % w).long()] = 1
                self._spu_mask.append(mask)

        # zero grad
        self.zero_grad(set_to_none=True)
        print("Done.")

    def _gradient_masking(self):
        """ Mask out unselected weights by directly modifying gradients """
        modules = self._localize()
        # print(modules[0].weight.data[100:108, 100:108])
        # print(self._spu_mask[0][100:108, 100:108])
        for module, mask in zip(modules, self._spu_mask):
            grad = module.weight.grad
            grad.detach().mul_(mask)

    def _localize(self) -> list[nn.Module]:
        """ Get specific modules, here we use first layer of MLP in transformer encoder """
        res = []
        assert isinstance(self.extractor._model, HubertCustom)
        for i in range(12):
            module = get_module_by_name(self.extractor._model, f"model.encoder.layers.{i}.fc1")  # nn.Linear
            res.append(module)
        return res

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
            self._gradient_masking()
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

    def training_step(self, batch, batch_idx):
        labels, _ = batch
        train_loss_dict, predictions, _ = self.common_step(batch, batch_idx, train=True)
        
        # print(self.global_step, batch_idx)
        self._custom_optimization(train_loss_dict["Total Loss"], batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions, '_batch': labels}
