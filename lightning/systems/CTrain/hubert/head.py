import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ClusterHead(nn.Module):
    def __init__(self, n_classes, d_in, dim, temperature=1.0) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, dim)
        self.table = nn.Parameter(torch.randn(n_classes, dim))
        self.temperature = temperature

        nn.init.uniform_(self.table)

    def forward(self, x):
        x = self.proj(x)  # B, L, dim
        embs = self.table.unsqueeze(0).unsqueeze(0)  # 1, 1, n_c, dim
        sim = F.cosine_similarity(embs, x.unsqueeze(2), dim=3) / self.temperature  # B, L, n_c

        return sim
    

class MultiClusterHead(nn.Module):
    def __init__(self, tid2n_cls: Dict[str, int], d_in, dim, temperature=1.0):
        super().__init__()
        self.tid2n_cls = tid2n_cls
        
        self.heads = nn.ModuleDict()
        for tid, n in tid2n_cls.items():
            self.heads[tid] = ClusterHead(n, d_in, dim, temperature)

    def forward(self, x, tid: str):
        return self.heads[tid](x)
