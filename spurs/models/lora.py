from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    last_n_layers: int = 3


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling



def _freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False



def apply_lora_to_saprot(backbone: nn.Module, cfg: LoRAConfig) -> None:
    _freeze_module(backbone)
    target_layers: Iterable[nn.Module] = backbone.layers[-cfg.last_n_layers:]
    for layer in target_layers:
        layer.attn.q_proj = LoRALinear(layer.attn.q_proj, cfg.rank, cfg.alpha, cfg.dropout)
        layer.attn.v_proj = LoRALinear(layer.attn.v_proj, cfg.rank, cfg.alpha, cfg.dropout)
        for p in layer.attn.q_proj.parameters():
            p.requires_grad = True
        for p in layer.attn.v_proj.parameters():
            p.requires_grad = True
