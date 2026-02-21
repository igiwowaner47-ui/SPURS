import torch
import torch.nn as nn


class RewiringAdapter(nn.Module):
    """LN -> CrossAttn -> Zero-init out proj -> Residual -> MLP -> Residual."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, mlp_ratio: float = 0.5):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        mlp_hidden = max(1, int(embed_dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context, attn_mask=None, key_padding_mask=None):
        residual = x
        x_norm = self.norm(x)
        attn_out, _ = self.cross_attn(
            query=x_norm,
            key=context,
            value=context,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + self.out_proj(attn_out)
        x = x + self.mlp(x)
        return x
