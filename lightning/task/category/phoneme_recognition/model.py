import torch.nn as nn
from typing import Dict

from dlhlp_lib.common.layers import WeightedSumLayer


class WSLinearModel(nn.Module):
    def __init__(self, n_in_layers:int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.ws = WeightedSumLayer(n_in_layers=n_in_layers)
        self.head = nn.Linear(d_in, d_out)

    def forward(self, x):
        # input: B, L, n_layers, dim
        x = self.ws(x, dim=2)  # B, L, dim
        return self.head(x)  # B, L, n_symbols
    

class WSBiLSTMModel(nn.Module):
    def __init__(self,
        n_in_layers:int, d_in: int, d_out: int,
        d_hidden: int=256,
        num_layers: int=2,
    ) -> None:
        super().__init__()
        self.ws = WeightedSumLayer(n_in_layers=n_in_layers)
        self.proj = nn.Linear(d_in, d_hidden)
        self.lstm = nn.LSTM(input_size=d_hidden, hidden_size=d_hidden // 2, 
                                num_layers=num_layers, bidirectional=True, batch_first=True)
        self.head = nn.Linear(d_hidden, d_out)

    def forward(self, x, lengths):
        """
        Args:
            x: Representation with shape (B, L, n_layers, d_in).
            lengths: Handle padding for LSTM.
        Return:
            Return tensor with shape (B, L, d_hidden)
        """
        x = self.ws(x, dim=2)  # B, L, d_in
        x = self.proj(x)  # B, L, d_hidden
        # total length should be record due to data parallelism issue (https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism)
        total_length = x.size(1)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)  # B, L, d_hidden
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=total_length)

        return self.head(x)  # B, L, n_symbols
