import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint

from dlhlp_lib.s3prl import S3PRLExtractor

from lightning.base.expert import BaseExpert, BaseMapper
from lightning.base.system import BaseSystem
from lightning.auto import AutoExpert


class System(BaseSystem):
    """ Multitask learning with hubert base. """

    experts: dict[str, BaseExpert]
    mappers: dict[str, BaseMapper]

    def __init__(self, config):
        super().__init__(config)

    def build_configs(self) -> None:
        self.task_configs = self.config["task_configs"]
        self.train_config = self.config["train_config"]
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

    def build_saver(self) -> list:
        checkpoint = ModelCheckpoint(
            dirpath=self.config["output_dir"]["ckpt_dir"],
            filename='{epoch}',
            monitor="Val/Total Loss", mode="min",
            save_top_k=-1
        )
        return [checkpoint]

    def training_step(self, batch, batch_idx):
        tid, batch = batch
        record = {"tid": tid}
        loss = self.mappers[tid].training_step(batch, batch_idx, record=record)

        # Log metrics to CometLogger
        # loss_dict = {f"Train/{k}": v.item() for k, v in record["losses"].items()}
        # self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': loss, 'record': record}
    
    def validation_step(self, batch, batch_idx):
        tid, batch = batch
        record = {"tid": tid}
        loss = self.mappers[tid].validation_step(batch, batch_idx, record=record)

        # Log metrics to CometLogger
        # loss_dict = {f"Val/{k}": v.item() for k, v in record["losses"].items()}
        # self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': loss, 'record': record}
    
    # refer to documents for multiple optimizers + schedulers
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
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
