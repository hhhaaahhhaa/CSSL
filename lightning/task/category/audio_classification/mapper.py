import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from lightning.base.expert import BaseMapper
from lightning.utils.tool import ssl_match_length
from ...model import WSUtteranceLevel


class ClassificationMapper(BaseMapper):
    """ CAUTION: this module is entangled with core class !!! """
    def __init__(self, mapper_config, core: nn.Module) -> None:
        super().__init__(mapper_config, core)
        self.classes = mapper_config["classes"]
        self.model = WSUtteranceLevel(
            n_in_layers=self.core.n_layers,
            d_in=self.core.dim,
            d_out=len(self.classes),
            **mapper_config["UtteranceLevel"]
        )
        self.loss_func = nn.CrossEntropyLoss()

    def build_optimized_model(self):
        return self.model
    
    def _common_step(self, batch, batch_idx, train=True):
        labels, repr_info = batch
        ssl_repr, ssl_repr_lens = self.core.extract(repr_info["wav"])
        ssl_repr = ssl_match_length(ssl_repr, target_len=max(ssl_repr_lens))
        output = self.model(ssl_repr, lengths=ssl_repr_lens)  # B, L, n_symbols

        loss = self.loss_func(output, labels["labels"])

        loss_dict = {
            "Total Loss": loss,
        }

        # pass length information
        info = {
            "ssl_repr_lens": torch.LongTensor(ssl_repr_lens)
        }
        return loss_dict, output, info
    
    def training_step(self, batch, batch_idx, record={}):
        train_loss_dict, predictions, info = self._common_step(batch, batch_idx, train=True)

        # log
        record["losses"] = train_loss_dict
        record["output"] = {
            "batch": batch,
            "predictions": predictions,
            "layer weight": self.get_layer_weight(),
        }

        return train_loss_dict["Total Loss"]

    def validation_step(self, batch, batch_idx, record={}):
        val_loss_dict, predictions, info = self._common_step(batch, batch_idx)

        # log
        record["losses"] = val_loss_dict
        record["output"] = defaultdict(list)

        # get class label
        labels, _ = batch
        predictions = predictions.argmax(dim=1).detach().cpu()
        record["output"]["gt"] = [self.classes[int(idx)] for idx in labels["labels"]]
        record["output"]["pred"] = [self.classes[int(idx)] for idx in predictions]

        return val_loss_dict["Total Loss"]

    def forward(self, x):
        pass
    
    @torch.no_grad
    def inference(self, wav: np.ndarray):
        #TODO
        pass

    @torch.no_grad
    def get_layer_weight(self):
        res = torch.nn.functional.softmax(self.model.ws.weight_raw, dim=0).data
        return res.detach().cpu()
