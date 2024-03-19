import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint

from dlhlp_lib.s3prl import S3PRLExtractor

from lightning.base.system import BaseSystem
from lightning.auto import AutoExpert


class System(BaseSystem):
    """ Naive single task learning with hubert base. """

    def __init__(self, config):
        super().__init__(config)

    def build_configs(self) -> None:
        self.task_configs = self.config["task_configs"]
        self.train_config = self.config["train_config"]
        self.algorithm_config = self.config["algorithm_config"]
        self.bs = self.train_config["optimizer"]["batch_size"]

        # build expert
        assert len(self.task_configs) == 1
        for tid, task_config in self.task_configs.items():
            self.tid = tid
            self.task_config = task_config
        self.expert = AutoExpert.from_config(self.task_config)
    
    def build_model(self) -> None:
        self.core = S3PRLExtractor("hubert")
        self.core._model.model.feature_grad_mult = 1.0  # unfreeze CNN encoder

        self.mapper = self.expert.get_mapper(self.core)

        core_ckpt_path = self.config.get("core_checkpoint_path", None)
        if core_ckpt_path is not None:
            self._load_core_checkpoint(core_ckpt_path)
        if self.algorithm_config is not None and "freeze_mode" in self.algorithm_config:
            self._set_freeze_mode(self.algorithm_config["freeze_mode"])

    def build_saver(self) -> list:
        checkpoint = ModelCheckpoint(
            dirpath=self.config["output_dir"]["ckpt_dir"],
            filename='{epoch}',
            monitor="Val/Total Loss", mode="min",
            save_top_k=-1
        )
        return [checkpoint]

    def training_step(self, batch, batch_idx):
        record = {}
        loss = self.mapper.training_step(batch, batch_idx, record=record)

        return {'loss': loss, 'record': record}
    
    def validation_step(self, batch, batch_idx):
        record = {}
        loss = self.mapper.validation_step(batch, batch_idx, record=record)

        return {'loss': loss, 'record': record}

    # refer to documents for multiple optimizers + schedulers
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        optimized_module = nn.ModuleList([self.core, self.mapper.build_optimized_model()])
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

    def _set_freeze_mode(self, mode: str):
        self.core.unfreeze()
        if mode == "all":
            self.core.freeze()
        elif mode == "none":
            pass
        elif mode == "cnn":
            cnn = self.core._model.model.feature_extractor
            for p in cnn.parameters():
                p.requires_grad = False
        elif mode == "!cnn":
            self.core.freeze()
            cnn = self.core._model.model.feature_extractor
            for p in cnn.parameters():
                p.requires_grad = True
        else:
            raise NotImplementedError

    def _load_core_checkpoint(self, core_ckpt_path: str):
        self.core.load_state_dict(torch.load(core_ckpt_path, map_location='cpu'))
