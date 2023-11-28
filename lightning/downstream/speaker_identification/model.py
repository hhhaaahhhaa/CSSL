import torch
import torch.nn as nn
from typing import Dict

from dlhlp_lib.common.layers import WeightedSumLayer


class MeanPooling(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, feature_BxTxH, features_len, **kwargs):
        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        agg_vec_list = []
        for i in range(len(feature_BxTxH)):
            agg_vec = torch.mean(feature_BxTxH[i][:features_len[i]], dim=0)
            agg_vec_list.append(agg_vec)

        return torch.stack(agg_vec_list)  # B, H


class LinearDownstream(nn.Module):
    def __init__(self, 
        n_in_layers:int,
        upstream_dim: int,
        d_out: int,
        d_hidden: int=256,
        use_proj: bool=True
    ) -> None:
        super().__init__()
        self.ws = WeightedSumLayer(n_in_layers=n_in_layers)
        self.d_hidden = d_hidden
        self.proj = nn.Linear(upstream_dim, self.d_hidden) if use_proj else None
        self.pooling = MeanPooling()
        if use_proj:
            self.head = nn.Linear(self.d_hidden, d_out)
        else:
            self.head = nn.Linear(upstream_dim, d_out)
    
    def forward(self, x, lengths):
        # input: B, L, n_layers, dim
        x = self.ws(x, dim=2)  # B, L, dim
        if self.proj is not None:
            x = self.proj(x)  # B, L, dim
        x = self.pooling(x, lengths)  # B, dim

        return self.head(x)  # B, d_out
