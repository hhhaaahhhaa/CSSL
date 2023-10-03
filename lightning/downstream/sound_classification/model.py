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
    def __init__(self, n_in_layers:int, upstream_dim: int, d_out: int) -> None:
        super().__init__()
        self.ws = WeightedSumLayer(n_in_layers=n_in_layers)
        self.pooling = MeanPooling()
        self.head = nn.Linear(upstream_dim, d_out)
    
    def forward(self, x, id: int):
        # input: B, L, n_layers, dim
        x = self.ws(x, dim=2)  # B, L, dim
        x = self.pooling(x)  # B, dim

        return self.head(x)  # B, d_out
