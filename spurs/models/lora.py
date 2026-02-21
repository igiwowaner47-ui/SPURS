from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 1.0
    target_layers: List[int] = field(default_factory=lambda: [-3, -2, -1])
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


class LoRALinear(nn.Module):
    """LoRA wrapper for Linear layers.

    Keeps the original linear frozen and learns a low-rank update BA.
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear only supports nn.Linear, got {type(base_layer)}")
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        self.reset_parameters()

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out



def _resolve_layer_indices(num_layers: int, target_layers: Sequence[int]) -> List[int]:
    resolved = []
    for layer_idx in target_layers:
        if layer_idx >= num_layers:
            # allow 1-based indexing: [L-2, L-1, L]
            layer_idx = layer_idx - 1
        layer_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
        if 0 <= layer_idx < num_layers:
            resolved.append(layer_idx)
    return sorted(set(resolved))


def inject_lora(
    layers: nn.ModuleList,
    rank: int,
    target_layers: Sequence[int],
    target_modules: Iterable[str],
    alpha: float = 1.0,
) -> List[str]:
    """Inject LoRA into `layers[layer].self_attn.{q_proj,v_proj}`.

    Returns injected module paths.
    """
    module_names = set(target_modules)
    injected = []

    for layer_idx in _resolve_layer_indices(len(layers), target_layers):
        layer = layers[layer_idx]
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            continue

        for module_name in module_names:
            if not hasattr(self_attn, module_name):
                continue
            module = getattr(self_attn, module_name)
            if isinstance(module, LoRALinear):
                continue
            if not isinstance(module, nn.Linear):
                continue
            setattr(self_attn, module_name, LoRALinear(module, rank=rank, alpha=alpha))
            injected.append(f"layers.{layer_idx}.self_attn.{module_name}")

    return injected
