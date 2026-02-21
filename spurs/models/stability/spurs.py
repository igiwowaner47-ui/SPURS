from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from spurs.models import register_model
from spurs.models.stability.basemodel import BaseModel
from spurs.models.stability.protein_mpnn import ProteinMPNNConfig

from spurs import utils
from spurs.models.stability.org_transfer_model import get_protein_mpnn
from spurs.models.stability.modules.esm2_adapter import ESM2WithStructuralAdatper
import torch.nn.functional as F
from spurs.models.stability.modules.esm2_adapter import StructuralAdapterConfig, StructuralAdapterStack
from spurs.models.saprot_backbone import SaProtBackbone, SaProtConfig
import torch.nn as nn
log = utils.get_logger(__name__)
from .mlp import MLPConfig
from spurs.models.ddg_head import DDGHead


@dataclass
class SPURSConfig:
    encoder: ProteinMPNNConfig = field(default=ProteinMPNNConfig())
    separate_loss: bool = True
    saprot_model_name: str = 'westlake-repl/SaProt_650M_AF2'
    freeze_saprot_backbone: bool = True
    use_lora_last_three_layers: bool = False
    dropout: float = 0.1
    mlp: MLPConfig = field(default=MLPConfig())


@register_model('spurs')
class SPURS(BaseModel):
    """
    SPURS (Structure-based Protein Understanding and Recognition System) model for protein stability prediction.
    
    This model combines protein structure information (from ProteinMPNN) and sequence information (from ESM2)
    to predict protein stability changes. The architecture consists of three main components:
    
    1. Encoder (ProteinMPNN): Processes protein structure information
    2. Decoder (ESM2): Processes sequence information with structural prior
    3. MLP: Final stability prediction layer
    
    The model uses a structural adapter to effectively combine structural and sequence information,
    allowing for more accurate stability predictions.
    
    Args:
        cfg (SPURSConfig): Configuration object containing model parameters
            - encoder: ProteinMPNN configuration
            - name: ESM2 model name
            - dropout: Dropout rate
            - mlp: MLP configuration
    """
    _default_cfg = SPURSConfig()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.tune = cfg.encoder.tune
        self.use_input_decoding_order = cfg.encoder.use_input_decoding_order
        self.encoder = get_protein_mpnn(tune=cfg.encoder.tune) 
        
        self.cfg.encoder.d_model = self.cfg.mlp.input_dim
        self.decoder = SaProtBackbone(SaProtConfig(
            model_name=self.cfg.saprot_model_name,
            freeze_backbone=self.cfg.freeze_saprot_backbone,
            use_lora_last_three_layers=self.cfg.use_lora_last_three_layers,
        ))
        self.structural_adapter = StructuralAdapterStack(StructuralAdapterConfig(
            embed_dim=1280,
            encoder_embed_dim=self.cfg.mlp.input_dim,
            dropout=self.cfg.dropout,
        ))
        self.input_dim = self.cfg.mlp.input_dim
        self.ddg_head = DDGHead(hidden_dim=1280, out_dim=20)
        
        self.padding_idx = self.decoder.padding_idx
        self.mask_idx = self.decoder.mask_idx
        self.cls_idx = self.decoder.cls_idx
        self.eos_idx = self.decoder.eos_idx
        
    def forward(self, batch, **kwargs):
        if not self.tune:
            with torch.no_grad():
                batch['feats'] = self.forward_encoder(batch)
        else:   
            batch['feats'] = self.forward_encoder(batch)
        
        batch['feats'] = batch['feats'][:,:,:self.input_dim]
        encoder_out = {'feats':F.pad(batch['feats'], (0, 0, 1, 1))}
        
        init_pred = batch['tokens']

        decoder_out = self.decoder(
            tokens=init_pred,
            encoder_out=encoder_out,
        )
        
        representation = decoder_out['representations'][-1]
        residue_representation = representation[:, 1:-1, :]
        residue_scores = self.ddg_head(residue_representation)
        base_feats = F.pad(batch['feats'], (0, 0, 1, 1))
        encoder_out = {
            'profam': F.pad(batch.get('profam_feats', batch['feats']), (0, 0, 1, 1)),
            'boltz2': F.pad(batch.get('boltz2_feats', batch['feats']), (0, 0, 1, 1)),
            'ligandmpnn': F.pad(batch.get('ligandmpnn_feats', batch['feats']), (0, 0, 1, 1)),
            'feats': base_feats,
        }
        
        init_pred = batch.get('saprot_tokens', batch['tokens'])

        decoder_out = self.decoder(tokens=init_pred)
        representation = self.structural_adapter(decoder_out['representations'][-1], encoder_out=encoder_out)
        if self.cfg.mlp.flat_dim > 0:
        
            flat_representation = self.flat_layers(representation)
            flat_representation = self.dp(flat_representation)
            flat_representation = F.gelu(flat_representation)
            representation = flat_representation
        
        representation = torch.cat([representation, encoder_out['feats']], dim=-1)

        if 'return_logist' in kwargs and kwargs['return_logist']:
            return residue_scores

        mut_pos = batch['mut_pos'].to(residue_scores.device)
        wt_aa_id = batch['wt_aa_id'].to(residue_scores.device)
        mut_aa_id = batch['mut_aa_id'].to(residue_scores.device)

        if mut_pos.dim() == 1:
            if residue_scores.size(0) == 1:
                mut_pos = mut_pos.unsqueeze(0)
                wt_aa_id = wt_aa_id.unsqueeze(0)
                mut_aa_id = mut_aa_id.unsqueeze(0)
            elif mut_pos.size(0) == residue_scores.size(0):
                mut_pos = mut_pos.unsqueeze(-1)
                wt_aa_id = wt_aa_id.unsqueeze(-1)
                mut_aa_id = mut_aa_id.unsqueeze(-1)

        batch_idx = torch.arange(residue_scores.size(0), device=residue_scores.device).unsqueeze(-1)
        score_mut = residue_scores[batch_idx, mut_pos.long(), mut_aa_id.long()]
        score_wt = residue_scores[batch_idx, mut_pos.long(), wt_aa_id.long()]
        delta = score_mut - score_wt

        return delta.reshape(-1)

    def forward_encoder(self,batch):
        """
        Forward pass through the encoder (ProteinMPNN) component of the SPURS model.
        
        This function processes the input protein structure data (X, S, mask, chain_M, residue_idx, chain_encoding_all, randn_1)
        and returns the encoded features from the ProteinMPNN encoder.

        Args:
            batch (dict): Input batch containing protein structure data
                - X: Protein structure coordinates
                - S: Protein structure mask
                - mask: Mask indicating valid positions
                - chain_M: Chain mask
                - residue_idx: Residue indices
                - chain_encoding_all: Chain encoding
        
        Returns:
            torch.Tensor: Encoded features from the ProteinMPNN encoder     
        """

        
        X = batch['X']
        S = batch['S']
        mask = batch['mask']
        chain_M = batch['chain_M']
        chain_M_chain_M_pos = batch['chain_M_chain_M_pos']
        residue_idx = batch['residue_idx']
        chain_encoding_all = batch['chain_encoding_all']
        randn_1 = batch['randn_1']
        all_mpnn_hid, mpnn_embed, _ = self.encoder(X, S, mask, chain_M, residue_idx, chain_encoding_all, None,self.use_input_decoding_order)
        
        all_mpnn_hid = torch.cat([all_mpnn_hid[0],mpnn_embed,all_mpnn_hid[1],all_mpnn_hid[2]],dim=-1)
        
        return all_mpnn_hid