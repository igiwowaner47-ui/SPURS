from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LigandMPNNEncoderConfig:
    hidden_dim: int = 256
    aa_vocab_size: int = 21
    input_dim: int = 3


class LigandMPNNEncoder(nn.Module):
    """Full-context joint graph encoder stub compatible with SPURS v6 interfaces."""

    def __init__(self, cfg: LigandMPNNEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.node_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.decoder_stack = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.aa_embedding = nn.Embedding(cfg.aa_vocab_size, cfg.hidden_dim)
        self.geo_proj = nn.Linear(cfg.hidden_dim * 2, 1280)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch.get('X')
        if x is None:
            raise KeyError("LigandMPNNEncoder requires batch['X']")
        if x.dim() == 4:
            node_feat = x.mean(dim=2)
        else:
            node_feat = x
        v_dec = self.decoder_stack(self.node_proj(node_feat))

        wt_ids = batch.get('wt_aa_id')
        if wt_ids is None:
            s = batch.get('S')
            if s is None:
                wt_ids = torch.zeros(v_dec.shape[:2], dtype=torch.long, device=v_dec.device)
            else:
                wt_ids = s.long()
        e_aa = self.aa_embedding(wt_ids)
        f_geo = torch.cat([v_dec, e_aa], dim=-1)
        return self.geo_proj(f_geo)
