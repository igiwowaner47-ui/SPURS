import torch.nn as nn


class DDGHead(nn.Module):
    """Per-residue ddG scoring head that outputs 20 amino-acid substitution scores."""

    def __init__(self, hidden_dim: int = 1280, out_dim: int = 20) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.proj(self.norm(x))
