from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
FOLDSEEK_TOKENS = list("abcdefghijklmnopqrstuvwxyz") + ["#"]


@dataclass
class SaProtConfig:
    model_name: str = "westlake-repl/SaProt_650M_AF2"
    hidden_size: int = 1280
    freeze_backbone: bool = True
    use_lora_last_three_layers: bool = False


class SaProtDualTokenizer:
    """Tokenizer for amino-acid + foldseek dual modality.

    Token format is `<AA>|<FS>` and special tokens are independent from ESM alphabets.
    """

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    MASK = "<mask>"
    UNK = "<unk>"

    def __init__(self):
        self.special_tokens = [self.PAD, self.BOS, self.EOS, self.MASK, self.UNK]
        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        for aa in AA_TOKENS:
            for fs in FOLDSEEK_TOKENS:
                token = f"{aa}|{fs}"
                self.vocab[token] = len(self.vocab)
        self.idx_to_tok = {i: t for t, i in self.vocab.items()}

        self.pad_idx = self.vocab[self.PAD]
        self.cls_idx = self.vocab[self.BOS]
        self.eos_idx = self.vocab[self.EOS]
        self.mask_idx = self.vocab[self.MASK]
        self.unk_idx = self.vocab[self.UNK]

    def _pair_to_token(self, aa: str, fs: str) -> str:
        aa = aa if aa in AA_TOKENS else "X"
        fs = fs if fs in FOLDSEEK_TOKENS else "#"
        tok = f"{aa}|{fs}"
        return tok if tok in self.vocab else self.UNK

    def encode(
        self,
        aa_seq: str,
        foldseek_seq: Optional[str] = None,
        add_special_tokens: bool = True,
        mlm_mask_positions: Optional[Sequence[int]] = None,
    ) -> List[int]:
        if foldseek_seq is None:
            foldseek_seq = "#" * len(aa_seq)
        if len(aa_seq) != len(foldseek_seq):
            raise ValueError("aa_seq and foldseek_seq must have the same length")

        ids = [self.vocab.get(self._pair_to_token(a, f), self.unk_idx) for a, f in zip(aa_seq, foldseek_seq)]
        if add_special_tokens:
            ids = [self.cls_idx] + ids + [self.eos_idx]

        if mlm_mask_positions is not None:
            offset = 1 if add_special_tokens else 0
            for pos in mlm_mask_positions:
                pos_with_offset = pos + offset
                if 0 <= pos_with_offset < len(ids):
                    ids[pos_with_offset] = self.mask_idx
        return ids

    def batch_encode(
        self,
        aa_seqs: Sequence[str],
        foldseek_seqs: Optional[Sequence[Optional[str]]] = None,
        mlm_masks: Optional[Sequence[Optional[Sequence[int]]]] = None,
    ) -> torch.Tensor:
        if foldseek_seqs is None:
            foldseek_seqs = [None] * len(aa_seqs)
        if mlm_masks is None:
            mlm_masks = [None] * len(aa_seqs)

        encoded = [
            self.encode(aa_seq=aa, foldseek_seq=fs, mlm_mask_positions=mask)
            for aa, fs, mask in zip(aa_seqs, foldseek_seqs, mlm_masks)
        ]
        max_len = max(len(x) for x in encoded)
        batch = torch.full((len(encoded), max_len), self.pad_idx, dtype=torch.long)
        for i, ids in enumerate(encoded):
            batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return batch


class SaProtBackbone(nn.Module):
    """SaProt backbone wrapper exposing unified hidden states (dim=1280)."""

    def __init__(self, cfg: SaProtConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = SaProtDualTokenizer()
        self.padding_idx = self.tokenizer.pad_idx
        self.mask_idx = self.tokenizer.mask_idx
        self.cls_idx = self.tokenizer.cls_idx
        self.eos_idx = self.tokenizer.eos_idx

        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "transformers is required for SaProtBackbone. Please install transformers."
            ) from e

        self.model = AutoModel.from_pretrained(cfg.model_name, trust_remote_code=True)
        if cfg.use_lora_last_three_layers:
            try:
                from peft import LoraConfig, get_peft_model
                lora_cfg = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                    layers_to_transform=list(range(max(0, self.model.config.num_hidden_layers - 3), self.model.config.num_hidden_layers)),
                )
                self.model = get_peft_model(self.model, lora_cfg)
            except Exception:
                # keep backbone usable even when peft is unavailable
                pass
        self.hidden_size = getattr(self.model.config, "hidden_size", cfg.hidden_size)
        if self.hidden_size != cfg.hidden_size:
            self.proj = nn.Linear(self.hidden_size, cfg.hidden_size)
            self.hidden_size = cfg.hidden_size
        else:
            self.proj = nn.Identity()

        if cfg.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Dict[int, torch.Tensor]]:
        if attention_mask is None:
            attention_mask = tokens.ne(self.padding_idx).long()
        outputs = self.model(input_ids=tokens, attention_mask=attention_mask, output_hidden_states=True)
        hidden = self.proj(outputs.last_hidden_state)
        return {"representations": {-1: hidden}, "hidden_states": hidden, "attention_mask": attention_mask}

    def tokenize(
        self,
        aa_seqs: Sequence[str],
        foldseek_seqs: Optional[Sequence[Optional[str]]] = None,
        mlm_masks: Optional[Sequence[Optional[Sequence[int]]]] = None,
    ) -> torch.Tensor:
        return self.tokenizer.batch_encode(aa_seqs, foldseek_seqs, mlm_masks)
