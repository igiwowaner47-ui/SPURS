from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BoltzEncoderConfig:
    input_dim: int = 256


class BoltzEncoder(nn.Module):
    def __init__(self, cfg: BoltzEncoderConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.input_dim, 1280)

    def forward(self, batch: dict, seq_len: int, device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            s = batch.get('boltz_single')
            if s is None:
                b = batch['tokens'].shape[0]
                s = torch.zeros(b, seq_len, self.proj.in_features, device=device)
            else:
                s = s.to(device)
        return self.proj(s)
