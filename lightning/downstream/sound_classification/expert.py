from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Dict

from lightning.base.system import System
from lightning.systems import load_system
from lightning.utils.tool import ssl_match_length
from .config_reader import ConfigReader
from .model import LinearDownstream
from .saver import Saver


class Expert(System):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_configs(self, upstream_info: Dict) -> None:
        self.data_configs = [ConfigReader.read(x) for x in self.data_configs]
        self.model_config = {  # self.model_config default is None in this task
            "upstream": upstream_info
        }
        self.bs = self.train_config["optimizer"]["batch_size"]
        self.classes = self.data_configs[0]["classes"]  # Currenly only support single dataset
    
    def build_model(self, upstream_info: Dict) -> None:        
        self.upstream = load_system(**upstream_info)
        self.upstream.freeze()
        self.model = LinearDownstream(
            n_in_layers=self.upstream.n_layers,
            upstream_dim=self.upstream.dim,
            d_out=len(self.classes)
        )
        self.loss_func = nn.CrossEntropyLoss()

    def build_optimized_model(self):
        return nn.ModuleList([self.model])

    def build_saver(self):
        saver = Saver(self.data_configs, self.log_dir, self.result_dir)
        return saver

    def common_step(self, batch, batch_idx, train=True):
        labels, repr_info = batch

        self.upstream.eval()
        with torch.no_grad():
            ssl_repr, ssl_repr_lens = self.upstream(repr_info["wav"])
            ssl_repr = ssl_match_length(ssl_repr, target_len=max(ssl_repr_lens))
            ssl_repr.detach()

        output = self.model.forward(ssl_repr, lengths=ssl_repr_lens)  # B, n_cls
        loss = self.loss_func(output, labels["labels"])

        loss_dict = {
            "Total Loss": loss,
        }
            
        return loss_dict, output, None
    
    def training_step(self, batch, batch_idx):
        labels, repr_info = batch
        train_loss_dict, predictions, _ = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions, '_batch': labels}
    
    def validation_step(self, batch, batch_idx):
        labels, repr_info = batch
        val_loss_dict, predictions, _ = self.common_step(batch, batch_idx)

        # calculate classification acc
        acc = (labels["labels"] == predictions.argmax(dim=1)).sum() / len(predictions)
        self.log_dict({"Val/Acc": acc}, sync_dist=True, batch_size=self.bs)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': labels, 'acc': acc}

    def on_save_checkpoint(self, checkpoint):
        """ (Hacking!) Remove pretrained weights in checkpoint to save disk space. """
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k in state_dict:
            if k.split('.')[0] == "upstream":
                continue
            new_state_dict[k] = state_dict[k]
        checkpoint["state_dict"] = new_state_dict

        return checkpoint
