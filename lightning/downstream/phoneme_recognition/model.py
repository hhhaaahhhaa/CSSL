import torch.nn as nn
from typing import Dict

from dlhlp_lib.common.layers import WeightedSumLayer


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
