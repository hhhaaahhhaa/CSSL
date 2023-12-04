import torch
import torch.nn as nn

from dlhlp_lib.s3prl import S3PRLExtractor

from lightning.base.system import System
from lightning.utils.tool import ssl_match_length
from lightning.systems.task_reader import TaskSequenceConfig, PlainConfigAdapter
from lightning.base.callbacks.saving import SavingScheduleCallback
from .config_reader import ConfigReader
from .model import HubertCustom
from .head import MultiClusterHead
from .saver import Saver


class HubertSystem(System):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_configs(self) -> None:
        # Compatible loading
        if "task_config" in self.data_configs[0]:
            self.task_config = TaskSequenceConfig(self.data_configs[0])
        else:
            self.task_config = PlainConfigAdapter(self.data_configs)
        self.data_configs = [ConfigReader.read(x) for x in self.task_config.get_tasks()]
        self.bs = self.train_config["optimizer"]["batch_size"]
    
    def build_model(self) -> None:
        self.extractor = S3PRLExtractor("hubert")
        self.extractor.set_model(HubertCustom())

        tid2n_cls = {}
        for conf in self.data_configs:
            tid2n_cls[conf["uid"]] = conf["n_symbols"]
        self.head = MultiClusterHead(
            tid2n_cls,
            d_in=self.extractor.dim,
            **self.model_config["head"],
        )
        
        self.loss_func = nn.CrossEntropyLoss(reduction="none")
        self.n_layers = self.extractor.n_layers
        self.dim = self.extractor.dim

    def build_optimized_model(self):
        return nn.ModuleList([self.extractor, self.head])

    def build_saver(self):
        saver = Saver(self.data_configs, self.log_dir, self.result_dir)
        saving_steps = self.task_config.get_info()["training"]["saving_steps"]
        if saving_steps:
            saving_schedule = SavingScheduleCallback(saving_steps)
            return [saver, saving_schedule]
        return saver

    def common_step(self, batch, batch_idx, train=True):
        labels, repr_info = batch

        assert isinstance(self.extractor._model, HubertCustom)
        res = self.extractor._model.mask_forward(repr_info["wav"], device=self.device)
        mask_indices = res["mask_indices"]

        uid = labels["uid"][0]  # Here we assume all uids in the same batch is identical
        sim = self.head(res["x"], tid=uid)  # B, L, n_c
        # print(labels["uid"])
        # print(sim.shape)

        # match length & only calculate loss on mask indices
        target = labels["codes"]  # B, L?
        with torch.no_grad():
            target = ssl_match_length(inputs=target.unsqueeze(-1), target_len=sim.shape[1]).squeeze(-1)  # B, L
        
        loss = self.loss_func(sim.transpose(1, 2), target)  # B, L
        loss = torch.mean(loss[mask_indices])
    
        loss_dict = {
            "Total Loss": loss,
        }

        return loss_dict, None, None

    def training_step(self, batch, batch_idx):
        labels, _ = batch
        train_loss_dict, predictions, _ = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions, '_batch': labels}

    def validation_step(self, batch, batch_idx):
        labels, _ = batch
        val_loss_dict, predictions, _ = self.common_step(batch, batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': labels}

    def forward(self, x):
        return self.extractor.extract(x)
