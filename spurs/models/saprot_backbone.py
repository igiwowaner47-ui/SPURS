from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


@dataclass
class SaProtBackboneConfig:
    hidden_dim: int = 1280
    num_layers: int = 33
    num_heads: int = 20
    aa_vocab_size: int = 33
    foldseek_vocab_size: int = 4096
    max_positions: int = 4096
    dropout: float = 0.1


class SaProtAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        b, n, c = x.shape
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask[:, None, None, :], float('-inf'))
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.out_proj(out)


class SaProtLayer(nn.Module):
    def __init__(self, cfg: SaProtBackboneConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = SaProtAttention(cfg.hidden_dim, cfg.num_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * 4, cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class SaProtBackbone(nn.Module):
    """Simplified SaProt-compatible frozen transformer backbone with dual token embeddings."""

    def __init__(self, cfg: SaProtBackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.aa_embedding = nn.Embedding(cfg.aa_vocab_size, cfg.hidden_dim)
        self.foldseek_embedding = nn.Embedding(cfg.foldseek_vocab_size, cfg.hidden_dim)
        self.position_embedding = nn.Embedding(cfg.max_positions, cfg.hidden_dim)
        self.layers = nn.ModuleList([SaProtLayer(cfg) for _ in range(cfg.num_layers)])
        self.final_ln = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        aa_tokens: torch.Tensor,
        foldseek_tokens: torch.Tensor,
        adapter_by_layer: Optional[Dict[int, tuple]] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, n = aa_tokens.shape
        pos = torch.arange(n, device=aa_tokens.device).unsqueeze(0).expand(b, n)
        x = self.aa_embedding(aa_tokens) + self.foldseek_embedding(foldseek_tokens) + self.position_embedding(pos)

        adapter_by_layer = adapter_by_layer or {}
        if token_mask is not None and token_mask.dtype is not torch.bool:
            token_mask = token_mask > 0
        hidden_states = {}
        for idx, layer in enumerate(self.layers):
            x = layer(x, attn_mask=token_mask)
            if idx in adapter_by_layer:
                adapter, f_ext = adapter_by_layer[idx]
                x = adapter(x, f_ext)
            hidden_states[idx + 1] = x

        x = self.final_ln(x)
        hidden_states[-1] = x
        return {"representations": hidden_states, "last_hidden_state": x}
