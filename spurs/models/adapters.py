from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RewiringAdapterConfig:
    hidden_dim: int = 1280
    num_heads: int = 16
    mlp_ratio: float = 2.0
    dropout: float = 0.1


class RewiringAdapter(nn.Module):
    """SPURS-style residual rewiring block using cross-attention."""

    def __init__(self, cfg: RewiringAdapterConfig):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.zero_out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        nn.init.zeros_(self.zero_out.weight)
        nn.init.zeros_(self.zero_out.bias)

        inner_dim = int(cfg.hidden_dim * cfg.mlp_ratio)
        self.mlp_ln = nn.LayerNorm(cfg.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(inner_dim, cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, h: torch.Tensor, f_ext: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        q = self.ln(h)
        attn_out, _ = self.cross_attn(q, f_ext, f_ext, key_padding_mask=key_padding_mask, need_weights=False)
        h_out = h + self.zero_out(attn_out)
        h_out = h_out + self.mlp(self.mlp_ln(h_out))
        return h_out
