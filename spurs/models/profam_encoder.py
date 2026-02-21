from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ProFamEncoderConfig:
    vocab_size: int = 33
    hidden_dim: int = 512


class ProFamEncoder(nn.Module):
    def __init__(self, cfg: ProFamEncoderConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.proj = nn.Linear(cfg.hidden_dim * 2, 1280)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        fwd = self.embedding(tokens)
        rev = torch.flip(self.embedding(torch.flip(tokens, dims=[1])), dims=[1])
        return self.proj(torch.cat([fwd, rev], dim=-1))
