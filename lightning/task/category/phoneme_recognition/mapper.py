import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchaudio.models.decoder import ctc_decoder
from collections import defaultdict

from lightning.base.expert import BaseMapper
from lightning.utils.tool import ssl_match_length
from .model import WSBiLSTMModel


class PRMapper(BaseMapper):
    """ CAUTION: this module is entangled with core class !!! """
    def __init__(self, mapper_config, core: nn.Module) -> None:
        super().__init__(mapper_config, core)
        self.tokens = mapper_config["tokens"]
        self.model = WSBiLSTMModel(
            n_in_layers=self.core.n_layers,
            d_in=self.core.dim,
            d_out=len(self.tokens),
            **mapper_config["lstm"]
        )
        self.loss_func = nn.CTCLoss(blank=1, zero_infinity=True)  # "-" is indexed as 1
        self.beam_search_decoder = ctc_decoder(
            lexicon=None,
            tokens=mapper_config["tokens"],
            lm=None,
            nbest=1,
            beam_size=50,
            beam_size_token=30
        )

    def build_optimized_model(self):
        return self.model
    
    def _common_step(self, batch, batch_idx, train=True):
        labels, repr_info = batch
        ssl_repr, ssl_repr_lens = self.core.extract(repr_info["wav"])
        ssl_repr = ssl_match_length(ssl_repr, target_len=max(ssl_repr_lens))
        output = self.model(ssl_repr, lengths=ssl_repr_lens)  # B, L, n_symbols

        # ctc
        log_probs = torch.log_softmax(output, dim=2)  # B, L, n_symbols
        loss = self.loss_func(log_probs.transpose(0, 1), labels["texts"], torch.LongTensor(ssl_repr_lens), labels["text_lens"].cpu())

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
            "emissions": torch.log_softmax(predictions.detach().cpu(), dim=2),
            "emission_lengths": info["ssl_repr_lens"]
        }

        return train_loss_dict["Total Loss"]
    
    def validation_step(self, batch, batch_idx, record={}):
        val_loss_dict, predictions, info = self._common_step(batch, batch_idx)

        # log
        record["losses"] = val_loss_dict
        record["output"] = defaultdict(list)

        # ctc decode
        labels, _ = batch
        emissions = torch.log_softmax(predictions.detach().cpu(), dim=2)
        beam_search_results = self.beam_search_decoder(emissions, info["ssl_repr_lens"])
        for idx, beams in enumerate(beam_search_results):
            # Dirty since we need to manipulate redundant silence token from self.beam_search_decoder!
            pred_transcript = self.beam_search_decoder.idxs_to_tokens(beams[0].tokens)
            pred_transcript = " ".join([p for p in pred_transcript if p != "|"])
            record["output"]["pred"].append(pred_transcript)
            
            gt_label = labels["texts"][idx][:labels["text_lens"][idx]].cpu()
            gt_transcript = self.beam_search_decoder.idxs_to_tokens(gt_label)
            gt_transcript = " ".join([p for p in gt_transcript if p != "|"])
            record["output"]["gt"].append(gt_transcript)
        
        return val_loss_dict["Total Loss"]

    def forward(self, x):
        pass
    
    @torch.no_grad
    def inference(self, wav: np.ndarray):
        ssl_repr, ssl_repr_lens = self.core.extract([wav])
        ssl_repr = ssl_match_length(ssl_repr, target_len=max(ssl_repr_lens))
        predictions = self.model(ssl_repr, lengths=ssl_repr_lens)  # B, L, n_symbols

        emissions = torch.log_softmax(predictions.detach().cpu(), dim=2)
        beam_search_results = self.beam_search_decoder(emissions, torch.LongTensor(ssl_repr_lens))
        beams = beam_search_results[0]
        # print(beams[0].tokens)
        pred_transcript = self.beam_search_decoder.idxs_to_tokens(beams[0].tokens)
        pred_transcript = " ".join([p for p in pred_transcript if p != "|"])

        return pred_transcript
