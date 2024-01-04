import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, d_in, d_out, dim=64):
        super().__init__()
        self.wa = nn.Linear(d_in, dim, bias=False)
        self.wb = nn.Linear(dim, d_out, bias=False)

    def forward(self, x):
        return self.wb(self.wa(x))
