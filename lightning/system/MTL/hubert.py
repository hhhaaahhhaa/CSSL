import torch
import torch.nn as nn

from dlhlp_lib.s3prl import S3PRLExtractor

from lightning.base.expert import BaseExpert, BaseMapper
from lightning.auto import AutoExpert
from .. import ONE


class System(ONE.hubert.System):
    """ Multitask learning with hubert base. """

    experts: dict[str, BaseExpert]
    mappers: dict[str, BaseMapper]

    def __init__(self, config):
        super().__init__(config)

    def build_configs(self) -> None:
        self.task_configs = self.config["task_configs"]
        self.train_config = self.config["train_config"]
        self.algorithm_config = self.config["algorithm_config"]
        self.bs = self.train_config["optimizer"]["batch_size"]

        # build experts
        self.experts = {}
        for tid, task_config in self.task_configs.items():
            self.experts[tid] = AutoExpert.from_config(task_config)
    
    def build_model(self) -> None:
        self.core = S3PRLExtractor("hubert")
        self.core._model.model.feature_grad_mult = 1.0  # unfreeze CNN encoder

        self.mappers = nn.ModuleDict()
        for tid in self.experts:
            self.mappers[tid] = self.experts[tid].get_mapper(self.core)

        core_ckpt_path = self.config.get("core_checkpoint_path", None)
        if core_ckpt_path is not None:
            self._load_core_checkpoint(core_ckpt_path)
        if self.algorithm_config is not None and "freeze_mode" in self.algorithm_config:
            self._set_freeze_mode(self.algorithm_config["freeze_mode"])

    def training_step(self, batch, batch_idx):
        tid, batch = batch
        record = {"tid": tid}
        loss = self.mappers[tid].training_step(batch, batch_idx, record=record)

        return {'loss': loss, 'record': record}
    
    def validation_step(self, batch, batch_idx):
        tid, batch = batch
        record = {"tid": tid}
        loss = self.mappers[tid].validation_step(batch, batch_idx, record=record)
        
        return {'loss': loss, 'record': record}
    
    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        optimized_modules = [module.build_optimized_model() for module in self.mappers.values()]
        optimized_module = nn.ModuleList([self.core, *optimized_modules])
        cnt = sum([p.numel() for p in optimized_module.parameters() if p.requires_grad])
        print(f"Optimizable parameters: {cnt}")

        optimizer = torch.optim.Adam(
            optimized_module.parameters(),
            lr=self.train_config["optimizer"]["lr"],
            betas=self.train_config["optimizer"]["betas"],
            eps=self.train_config["optimizer"]["eps"],
            weight_decay=self.train_config["optimizer"]["weight_decay"],
        )
        return optimizer
