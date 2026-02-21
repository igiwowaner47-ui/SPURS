import torch
import torch.nn as nn


class ProFAMEncoder(nn.Module):
    """Encode ProFAM forward/reverse streams and project to 1280 dims."""

    def __init__(self, out_dim: int = 1280):
        super().__init__()
        # input dim is 2d after concat(forward, reverse_aligned)
        self.proj = nn.LazyLinear(out_dim)

    @staticmethod
    def _masked_reverse(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reverse valid tokens only and keep padded positions untouched."""
        if mask is None:
            return torch.flip(x, dims=[1])

        x_aligned = x.clone()
        for b in range(x.size(0)):
            valid = mask[b].bool()
            if valid.any():
                x_aligned[b, valid] = torch.flip(x[b, valid], dims=[0])
        return x_aligned

    def forward(
        self,
        forward_repr: torch.Tensor,
        reverse_repr: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if reverse_repr is None:
            reverse_repr = self._masked_reverse(forward_repr, mask)
        else:
            reverse_repr = self._masked_reverse(reverse_repr, mask)

        fused = torch.cat([forward_repr, reverse_repr], dim=-1)
        return self.proj(fused)
