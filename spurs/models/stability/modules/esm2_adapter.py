from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
# https://github.com/BytedProtein/ByProt/blob/dd279dc85f76ee2c28c819b71bf3911b90159f0a/src/byprot/models/fixedbb/lm_design/modules/esm2_adapter.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Union

import torch
import torch.nn as nn

import esm
from esm.modules import (
    TransformerLayer,
    RobertaLMHead,
    ESM1bLayerNorm,
    ContactPredictionHead,
)
from spurs.models.adapters import RewiringAdapter
from spurs.utils.config import compose_config as Cfg, merge_config
from spurs import utils
log = utils.get_logger(__name__)

class ESM2WithStructuralAdatper(nn.Module):
    EXTERNAL_DIM = 1280
    ADAPTER_STREAM_ORDER = ("profam", "boltz2", "ligandmpnn")

    @classmethod
    def from_pretrained(cls, args, override_args=None, name='esm2_t33_650M_UR50D'):
        import esm
        pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet_hub(name)

        pretrained_args = Cfg(
            num_layers=pretrained_model.num_layers, 
            embed_dim=pretrained_model.embed_dim, 
            attention_heads=pretrained_model.attention_heads, 
            token_dropout=pretrained_model.token_dropout, 
        )
        args = merge_config(pretrained_args, args)
        model = cls(args, deepcopy(alphabet)) 
        out = model.load_state_dict(pretrained_model.state_dict(), strict=False)        
        log.info(f"missing keys: {out.missing_keys}")
        log.info(f"unexpected keys: {out.unexpected_keys}")
        del pretrained_model

        # freeze pretrained parameters
        for pname, param in model.named_parameters():
            if 'adapter' not in pname:
                param.requires_grad = False
        return model 

    def __init__(
        self,
        args,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
        # num_layers: int = 33,
        # embed_dim: int = 1280,
        # attention_heads: int = 20,
        # token_dropout: bool = True,
    ):
        super().__init__()
        self.args = args
        self.num_layers = args.num_layers
        self.embed_dim = args.embed_dim
        self.attention_heads = args.attention_heads
        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = args.token_dropout
        #CHANGED
        self.use_adapter = True
        self._init_submodules()
        
        if not self.use_adapter:
            for param in self.parameters():
                param.requires_grad = False
        
        

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )
        self.embed_tokens.eval()

        self.layers = nn.ModuleList(
            [
                self._init_layer(_)
                for _ in range(self.num_layers)
            ]
        )

        self.adapter_layer_to_stream = {
            self.num_layers - 3: "profam",
            self.num_layers - 2: "boltz2",
            self.num_layers - 1: "ligandmpnn",
        }
        self.rewiring_adapters = nn.ModuleDict(
            {
                stream: RewiringAdapter(
                    embed_dim=self.embed_dim,
                    num_heads=self.attention_heads,
                    dropout=self.args.dropout,
                )
                for stream in self.ADAPTER_STREAM_ORDER
            }
        )
        self.feature_projs = nn.ModuleDict(
            {
                stream: nn.LazyLinear(self.EXTERNAL_DIM)
                for stream in self.ADAPTER_STREAM_ORDER
            }
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        
        self.contact_head.eval()
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)
        self.emb_layer_norm_after.eval()
        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )
        self.lm_head.eval()

    def _init_layer(self, layer_idx):
        layer = TransformerLayer(
            self.embed_dim,
            4 * self.embed_dim,
            self.attention_heads,
            add_bias_kv=False,
            use_esm1b_layer_norm=True,
            use_rotary_embeddings=True,
        )
        return layer

    def _select_stream_features(self, encoder_out, stream):
        aliases = {
            "profam": ["profam", "profam_feats"],
            "boltz2": ["boltz2", "boltz2_feats"],
            "ligandmpnn": ["ligandmpnn", "ligandmpnn_feats", "feats"],
        }
        for key in aliases[stream]:
            value = encoder_out.get(key)
            if value is not None:
                return value
        raise KeyError(f"Missing required encoder stream '{stream}' in encoder_out.")

    def _project_stream_features(self, encoder_out):
        projected = {}
        for stream in self.ADAPTER_STREAM_ORDER:
            feats = self._select_stream_features(encoder_out, stream)
            projected[stream] = self.feature_projs[stream](feats)
        return projected

    def forward_layers(self, x, encoder_out, padding_mask, repr_layers=[], hidden_representations=[], need_head_weights=False, attn_weights=[]):
        stream_features = self._project_stream_features(encoder_out)
        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
            )
            stream = self.adapter_layer_to_stream.get(layer_idx)
            if stream is not None:
                adapter_context = stream_features[stream].transpose(0, 1)
                x = self.rewiring_adapters[stream](
                    x,
                    adapter_context,
                    attn_mask=None,
                    key_padding_mask=padding_mask,
                )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        return x, hidden_representations, attn_weights, layer_idx


    def forward(self, tokens, encoder_out, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]


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
        if not padding_mask.any():
            padding_mask = None

        # for layer_idx, layer in enumerate(self.layers):
        #     x, attn = layer(
        #         x,
        #         self_attn_padding_mask=padding_mask,
        #         need_head_weights=need_head_weights,
        #     )
        #     if (layer_idx + 1) in repr_layers:
        #         hidden_representations[layer_idx + 1] = x.transpose(0, 1)
        #     if need_head_weights:
        #         # (H, B, T, T) => (B, H, T, T)
        #         attn_weights.append(attn.transpose(1, 0))

        x, hidden_representations, attn_weights, layer_idx = self.forward_layers(
            x, encoder_out, padding_mask, 
            repr_layers=repr_layers, 
            hidden_representations=hidden_representations,
            need_head_weights=need_head_weights,
            attn_weights=attn_weights if need_head_weights else None
        )


        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        hidden_representations[-1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]
