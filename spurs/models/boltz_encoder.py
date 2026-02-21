import torch
import torch.nn as nn


class BoltzEncoder(nn.Module):
    """Extract Boltz single representation S and project to 1280 dims."""

    def __init__(self, out_dim: int = 1280):
        super().__init__()
        self.proj = nn.LazyLinear(out_dim)

    def _extract_single(self, boltz_out):
        if isinstance(boltz_out, torch.Tensor):
            return boltz_out
        if isinstance(boltz_out, dict):
            for key in ("single_repr", "single", "S"):
                if key in boltz_out:
                    return boltz_out[key]
        raise ValueError("Boltz output must contain single representation S [B,N,d_boltz].")

    def forward(self, boltz_model_or_repr, model_inputs=None):
        with torch.no_grad():
            if isinstance(boltz_model_or_repr, torch.Tensor) or isinstance(boltz_model_or_repr, dict):
                single_repr = self._extract_single(boltz_model_or_repr)
            else:
                if model_inputs is None:
                    boltz_out = boltz_model_or_repr()
                elif isinstance(model_inputs, dict):
                    boltz_out = boltz_model_or_repr(**model_inputs)
                elif isinstance(model_inputs, (tuple, list)):
                    boltz_out = boltz_model_or_repr(*model_inputs)
                else:
                    boltz_out = boltz_model_or_repr(model_inputs)
                single_repr = self._extract_single(boltz_out)

        return self.proj(single_repr)
