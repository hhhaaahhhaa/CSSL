import torch
import torch.nn as nn
import torch.nn.functional as F

from dlhlp_lib.s3prl import S3PRLExtractor

from lightning.base.system import System
from .config_reader import ConfigReader
from .model import HubertCustom
from .saver import Saver


class ClusterTable(nn.Module):
    def __init__(self, n_classes, dim) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.randn(n_classes, dim))
        nn.init.uniform_(self.table)

    def forward(self, x):
        return self.table(x)
    

class HubertSystem(System):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_configs(self) -> None:
        self.data_configs = [ConfigReader.read(x) for x in self.data_configs]
        self.model_config["head"]["n_classes"] = self.data_configs[0]["n_symbols"]
        self.bs = self.train_config["optimizer"]["batch_size"]
    
    def build_model(self) -> None:
        self.extractor = S3PRLExtractor("hubert")
        self.extractor.set_model(HubertCustom())
        self.final_proj = nn.Linear(self.extractor.dim, self.model_config["head"]["dim"])
        self.embs = ClusterTable(
            n_classes=self.model_config["head"]["n_classes"],
            dim=self.model_config["head"]["dim"]
        )
        
        self.loss_func = nn.CrossEntropyLoss(reduction="none")
        self.n_layers = self.extractor.n_layers
        self.dim = self.extractor.dim

    def build_optimized_model(self):
        return nn.ModuleList([self.extractor, self.final_proj, self.embs])

    def build_saver(self):
        saver = Saver(self.data_configs, self.log_dir, self.result_dir)
        return saver

    def common_step(self, batch, batch_idx, train=True):
        labels, repr_info = batch

        assert isinstance(self.extractor._model, HubertCustom)
        res = self.extractor._model.mask_forward(repr_info["wav"], device=self.device)
        # padding_mask = res["padding_mask"]
        mask_indices = res["mask_indices"]
        output = self.final_proj(res["x"])  # B, L, dim
        
        embs = self.embs.table.unsqueeze(0).unsqueeze(0)  # 1, 1, n_c, dim
        sim = F.cosine_similarity(embs, output.unsqueeze(2), dim=3) / self.model_config["head"]["temperature"]  # B, L, n_c
        # print(sim.shape)

        # match length & only calculate loss on mask indices
        target = labels["clusters"]  # B, L?
        if target.shape[1] > sim.shape[1]:
            target = target[:, :sim.shape[1]]  # (B, L)
        if target.shape[1] < sim.shape[1]:
            diff = sim.shape[1] - target.shape[1]
            pad_vec = target[:, -1].unsqueeze(1)  # (B, 1)
            target = torch.cat((target, pad_vec.repeat(1, diff)), dim=1)  # (B, L)
        # print(target.shape)
        loss = self.loss_func(sim.transpose(1, 2), target)  # B, L
        loss = torch.mean(loss[mask_indices])
    
        loss_dict = {
            "Total Loss": loss,
        }

        return loss_dict, output, None

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
