from collections import OrderedDict
import torch
import torch.nn as nn
from torchaudio.models.decoder import ctc_decoder
from typing import Dict
import jiwer

from dlhlp_lib.common.layers import WeightedSumLayer

from lightning.systems.system import System
from lightning.systems import load_system
from lightning.utils.tool import ssl_match_length
from .config_reader import ConfigReader
from .saver import Saver


class LinearDownstream(nn.Module):
    def __init__(self, n_in_layers:int, upstream_dim: int, id2symbols: Dict) -> None:
        super().__init__()
        self.ws = WeightedSumLayer(n_in_layers=n_in_layers)
        self.heads = nn.ModuleDict()
        for id, v in id2symbols.items():
            if len(v) > 0:
                self.heads[f"head-{id}"] = nn.Linear(upstream_dim, len(v))

    def forward(self, x, id: int):
        # input: B, L, n_layers, dim
        x = self.ws(x, dim=2)  # B, L, dim

        return self.heads[f"head-{id}"](x)  # B, L, n_symbols
    

class BiLSTMDownstream(nn.Module):
    """ BiLSTM """
    def __init__(self,
        n_in_layers:int, upstream_dim: int, id2symbols: Dict,
        d_hidden: int=256,
        num_layers: int=2,
        use_proj: bool=True,
    ) -> None:
        super().__init__()
        self.ws = WeightedSumLayer(n_in_layers=n_in_layers)
        self.d_hidden = d_hidden
        self.proj = nn.Linear(upstream_dim, self.d_hidden) if use_proj else None
        
        self.lstm = nn.LSTM(input_size=self.d_hidden, hidden_size=self.d_hidden // 2, 
                                num_layers=num_layers, bidirectional=True, batch_first=True)
        self.heads = nn.ModuleDict()
        for id, v in id2symbols.items():
            if len(v) > 0:
                if use_proj:
                    self.heads[f"head-{id}"] = nn.Linear(self.d_hidden, len(v))
                else:
                    self.heads[f"head-{id}"] = nn.Linear(upstream_dim, len(v))

    def forward(self, x, lengths, id: int):
        """
        Args:
            x: Representation with shape (B, L, n_layers, d_in).
            lengths: Handle padding for LSTM.
        Return:
            Return tensor with shape (B, L, d_hidden)
        """
        x = self.ws(x, dim=2)  # B, L, d_in
        if self.proj is not None:
            x = self.proj(x)  # B, L, d_hidden
        # total length should be record due to data parallelism issue (https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism)
        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)  # B, L, d_hidden
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=total_length)

        return self.heads[f"head-{id}"](x)  # B, L, n_symbols


class Expert(System):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_configs(self, upstream_info: Dict) -> None:
        self.data_configs = [ConfigReader.read(x) for x in self.data_configs]
        self.model_config.update({
            "upstream": upstream_info
        })
    
    def build_model(self, upstream_info: Dict) -> None:
        from text.define import LANG_ID2SYMBOLS

        self.upstream = load_system(**upstream_info)
        self.upstream.freeze()
        lang_id = self.data_configs[0]["lang_id"]  # Currenly only support single language
        id2symbols = {lang_id: LANG_ID2SYMBOLS[lang_id]}
        self.model = BiLSTMDownstream(
            n_in_layers=self.upstream.n_layers,
            upstream_dim=self.upstream.dim,
            id2symbols=id2symbols,
            d_hidden=self.model_config["lstm"]["d_hidden"],
            num_layers=self.model_config["lstm"]["num_layers"]
        )
        
        self.loss_func = nn.CTCLoss(blank=1, zero_infinity=True)  # "-" is indexed as 1
        self.beam_search_decoder = ctc_decoder(
            lexicon=None,
            tokens=id2symbols[lang_id],
            lm=None,
            nbest=1,
            beam_size=50,
            beam_size_token=30
        )

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

        print(ssl_repr.shape)
        print(ssl_repr_lens)
        input()

        output = self.model.forward(ssl_repr, lengths=ssl_repr_lens, id=repr_info["lang_id"])  # B, L, n_symbols

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

    def training_step(self, batch, batch_idx):
        labels, repr_info = batch
        train_loss_dict, predictions, _ = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': predictions, '_batch': labels, 'lang_id': repr_info["lang_id"]}

    
    def validation_step(self, batch, batch_idx):
        labels, repr_info = batch
        val_loss_dict, predictions, info = self.common_step(batch, batch_idx)

        # ctc decode to calculate acc
        emissions = torch.log_softmax(predictions.detach().cpu(), dim=2)
        beam_search_results = self.beam_search_decoder(emissions, info["ssl_repr_lens"])
        acc = 0
        for i, res in enumerate(beam_search_results):
            # Dirty since we need to manipulate redundant silence token from self.beam_search_decoder!
            pred_transcript = self.beam_search_decoder.idxs_to_tokens(res[0].tokens)
            pred_transcript = " ".join([p for p in pred_transcript if p != "|"])
            gt_transcript = self.beam_search_decoder.idxs_to_tokens(labels["texts"][i][:labels["text_lens"][i]].cpu())
            gt_transcript = " ".join([p for p in gt_transcript if p != "|"])
            acc += 1 - jiwer.wer(gt_transcript, pred_transcript)
        acc /= len(beam_search_results)
        self.log_dict({"Val/Acc": acc}, sync_dist=True)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': labels, 'lang_id': repr_info["lang_id"]}

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
