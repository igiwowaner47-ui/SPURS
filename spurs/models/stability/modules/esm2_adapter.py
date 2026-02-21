from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class StructuralAdapterConfig:
    embed_dim: int = 1280
    encoder_embed_dim: int = 128
    attention_heads: int = 8
    dropout: float = 0.1


class StructuralAdapterLayer(nn.Module):
    """Reusable structural adapter independent from ESM-specific classes."""

    def __init__(self, cfg: StructuralAdapterConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.attention_heads,
            kdim=cfg.encoder_embed_dim,
            vdim=cfg.encoder_embed_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim // 2, cfg.embed_dim),
        )

    def forward(self, hidden_states: torch.Tensor, encoder_feats: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(hidden_states)
        attn_out, _ = self.cross_attn(
            query=x,
            key=encoder_feats,
            value=encoder_feats,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = hidden_states + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class StructuralAdapterStack(nn.Module):
    def __init__(self, cfg: StructuralAdapterConfig):
        super().__init__()
        self.layer = StructuralAdapterLayer(cfg)

    def forward(self, hidden_states: torch.Tensor, encoder_out: Optional[dict] = None) -> torch.Tensor:
        if encoder_out is None or 'feats' not in encoder_out:
            return hidden_states
        encoder_feats = encoder_out['feats']
        return self.layer(hidden_states, encoder_feats)


# backward-compatible alias used by existing imports
ESM2WithStructuralAdatper = StructuralAdapterStack
