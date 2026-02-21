from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DDGHeadConfig:
    hidden_dim: int = 1280
    num_aa: int = 20


class DDGHead(nn.Module):
    def __init__(self, cfg: DDGHeadConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_aa)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(h))


def delta_from_logits(logits: torch.Tensor, mut_pos: torch.Tensor, wt_aa: torch.Tensor, mut_aa: torch.Tensor) -> torch.Tensor:
    if logits.shape[0] == 1 and mut_pos.dim() == 1 and mut_pos.numel() > 1:
        local = logits[0, mut_pos.long()]
        return local[torch.arange(mut_pos.numel(), device=logits.device), mut_aa.long()] - local[
            torch.arange(mut_pos.numel(), device=logits.device), wt_aa.long()
        ]

    b = logits.shape[0]
    idx = torch.arange(b, device=logits.device)
    local = logits[idx, mut_pos.long()]
    return local[idx, mut_aa.long()] - local[idx, wt_aa.long()]
