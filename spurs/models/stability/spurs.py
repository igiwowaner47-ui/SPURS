from dataclasses import dataclass, field

import torch
from spurs.models import register_model
from spurs.models.stability.basemodel import BaseModel

from spurs.models.adapters import RewiringAdapter, RewiringAdapterConfig
from spurs.models.boltz_encoder import BoltzEncoder, BoltzEncoderConfig
from spurs.models.ddg_head import DDGHead, DDGHeadConfig, delta_from_logits
from spurs.models.ligandmpnn_encoder import LigandMPNNEncoder, LigandMPNNEncoderConfig
from spurs.models.lora import LoRAConfig, apply_lora_to_saprot
from spurs.models.profam_encoder import ProFamEncoder, ProFamEncoderConfig
from spurs.models.saprot_backbone import SaProtBackbone, SaProtBackboneConfig


@dataclass
class SPURSConfig:
    saprot: SaProtBackboneConfig = field(default_factory=SaProtBackboneConfig)
    ligandmpnn: LigandMPNNEncoderConfig = field(default_factory=LigandMPNNEncoderConfig)
    profam: ProFamEncoderConfig = field(default_factory=ProFamEncoderConfig)
    boltz: BoltzEncoderConfig = field(default_factory=BoltzEncoderConfig)
    adapter: RewiringAdapterConfig = field(default_factory=RewiringAdapterConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    ddg_head: DDGHeadConfig = field(default_factory=DDGHeadConfig)


@register_model('spurs')
class SPURS(BaseModel):
    _default_cfg = SPURSConfig()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.backbone = SaProtBackbone(self.cfg.saprot)
        self.profam_encoder = ProFamEncoder(self.cfg.profam)
        self.boltz_encoder = BoltzEncoder(self.cfg.boltz)
        self.ligand_encoder = LigandMPNNEncoder(self.cfg.ligandmpnn)

        self.adapter_l_minus_2 = RewiringAdapter(self.cfg.adapter)
        self.adapter_l_minus_1 = RewiringAdapter(self.cfg.adapter)
        self.adapter_l = RewiringAdapter(self.cfg.adapter)

        self.ddg_head = DDGHead(self.cfg.ddg_head)

        apply_lora_to_saprot(self.backbone, self.cfg.lora)

    def _extract_wt_mut(self, batch, device):
        if 'wt_aa_id' in batch and 'mut_aa_id' in batch:
            return batch['wt_aa_id'].to(device), batch['mut_aa_id'].to(device)

        append = batch['append_tensors'].to(device)
        if append.dim() == 1:
            append = append.unsqueeze(0)
        wt_aa = torch.argmax(append[:, :21], dim=-1).clamp_max(19)
        mut_aa = torch.argmax(append[:, 21:], dim=-1).clamp_max(19)
        return wt_aa, mut_aa

    def forward(self, batch, **kwargs):
        aa_tokens = batch['tokens'].long()
        foldseek_tokens = batch.get('foldseek_tokens', torch.zeros_like(aa_tokens)).long().to(aa_tokens.device)

        f_profam = self.profam_encoder(aa_tokens)
        f_boltz = self.boltz_encoder(batch, aa_tokens.shape[1], aa_tokens.device)
        f_geo = self.ligand_encoder(batch)

        l = len(self.backbone.layers)
        adapter_by_layer = {
            l - 3: (self.adapter_l_minus_2, f_profam),
            l - 2: (self.adapter_l_minus_1, f_boltz),
            l - 1: (self.adapter_l, f_geo),
        }

        outputs = self.backbone(
            aa_tokens=aa_tokens,
            foldseek_tokens=foldseek_tokens,
            adapter_by_layer=adapter_by_layer,
            token_mask=batch.get('mask', None),
        )
        logits = self.ddg_head(outputs['last_hidden_state'])

        mut_pos = batch['mut_ids'] if isinstance(batch['mut_ids'], torch.Tensor) else torch.tensor(batch['mut_ids'])
        mut_pos = mut_pos.to(logits.device).long()
        wt_aa, mut_aa = self._extract_wt_mut(batch, logits.device)

        delta = delta_from_logits(logits, mut_pos, wt_aa.long(), mut_aa.long())
        delta[torch.isnan(delta)] = 10000
        return delta
