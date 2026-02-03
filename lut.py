import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional, Tuple

class KeyedLUTSampler3D_HardGumbel_Complex(nn.Module):
    """
    LUT per wavelength key: (K,H,W) complex/real
      - K = number of meta-atom libraries (orderless classes)

    grid supports two formats:
      1) per-library-xy: (B,2K,H_out,W_out) = (x0,y0,x1,y1,...,x_{K-1},y_{K-1})
      2) legacy:        (B,2K+1,H_out,W_out) 마지막 1채널은 무시 (호환용)

    class selection (ORDERLESS):
      - class_logits: (B,K,H_out,W_out)  -> softmax(tau) -> weights
      - use_hard=True면 STE hard forward(=one-hot) / soft backward

    returns:
      outputs: {key: (B,1,H_out,W_out)} complex/real (true complex mixture)
    """
    def __init__(
        self,
        lut_dict: Dict[str, torch.Tensor],
        align_corners: bool = True,
        tau: float = 1.0,
        use_hard: bool = False,
        hard_eval: bool = True,              # eval()에서도 use_hard면 hard 유지
        eps: float = 1e-8,
    ):
        super().__init__()
        assert len(lut_dict) > 0, "lut_dict must not be empty."
        self.align_corners = bool(align_corners)
        self.tau = float(tau)
        self.use_hard = bool(use_hard)
        self.hard_eval = bool(hard_eval)

        self.eps = float(eps)

        self.keys = []
        for k, lut in lut_dict.items():
            assert lut.ndim in (2, 3), f"LUT '{k}' must be (H,W) or (K,H,W)."
            if lut.ndim == 2:
                lut = lut.unsqueeze(0)  # (H,W)->(1,H,W)
            self.register_buffer(f"lut_vol__{k}", lut.contiguous(), persistent=False)
            self.keys.append(k)

    # -------------------------
    # utilities
    # -------------------------
    @staticmethod
    def _gumbel_like(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-9
        u = torch.rand_like(x).clamp_(eps, 1 - eps)
        return -torch.log(-torch.log(u))

    def _weights_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,K,H,W)
        returns weights: (B,K,H,W)
        """
        soft = F.softmax(logits / max(self.tau, self.eps), dim=1)

        want_hard = self.use_hard and (self.training or self.hard_eval)
        if not want_hard:
            return soft

        # STE hard
        with torch.no_grad():
            top = torch.argmax(soft, dim=1, keepdim=True)          # (B,1,H,W)
            hard = torch.zeros_like(soft).scatter_(1, top, 1.0)    # (B,K,H,W)
        return (hard - soft).detach() + soft

    def _parse_grid(self, grid: torch.Tensor, K: int) -> torch.Tensor:
        """
        grid -> (B,K,H,W,2) where last dim is (x,y) in [-1,1]
        accepts:
          (B,2K,H,W) or (B,2K+1,H,W) (last ignored)
        """
        assert grid.ndim == 4, "grid must be (B,C,H,W)"
        B, Cg, H, W = grid.shape
        if Cg == 2 * K:
            xy = grid
        elif Cg == 2 * K + 1:
            xy = grid[:, :2*K]
        else:
            raise ValueError(f"grid channel must be 2K or 2K+1, got {Cg} (K={K})")

        xy = xy.view(B, K, 2, H, W).permute(0, 1, 3, 4, 2).contiguous()  # (B,K,H,W,2)
        return xy

    # -------------------------
    # forward
    # -------------------------
    def forward(
        self,
        keys: Optional[Iterable[str]],
        grid: torch.Tensor,
        class_logits: torch.Tensor,   # (B,K,H,W)
    ) -> Dict[str, torch.Tensor]:

        if keys is None:
            keys = self.keys
        else:
            keys = list(keys)

        # infer K from LUT
        lut0 = getattr(self, f"lut_vol__{keys[0]}")
        K, H_lut, W_lut = lut0.shape

        assert class_logits.ndim == 4 and class_logits.shape[1] == K, \
            f"class_logits must be (B,{K},H,W), got {tuple(class_logits.shape)}"

        B, _, H_out, W_out = class_logits.shape

        # grid -> (B,K,H_out,W_out,2)
        grid_xy_all = self._parse_grid(grid, K=K)  # (B,K,H_out,W_out,2)
        assert grid_xy_all.shape[2] == H_out and grid_xy_all.shape[3] == W_out, \
            "grid spatial size must match class_logits spatial size"

        weights = self._weights_from_logits(class_logits)          # (B,K,H,W)
        w = weights.unsqueeze(2)                                   # (B,K,1,H,W)

        outputs: Dict[str, torch.Tensor] = {}

        # sample per wavelength key
        for kk in keys:
            lut = getattr(self, f"lut_vol__{kk}")  # (K,H_lut,W_lut), complex/real
            assert lut.shape[0] == K, "LUT K mismatch."

            # (B*K,1,H_lut,W_lut)
            inp = lut.unsqueeze(1).unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B,K,1,H,W)
            inp = inp.reshape(B * K, 1, H_lut, W_lut)

            # (B*K,H_out,W_out,2)
            grid_rep = grid_xy_all.reshape(B * K, H_out, W_out, 2)

            if torch.is_complex(inp):
                real = F.grid_sample(inp.real, grid_rep, mode="bilinear", align_corners=self.align_corners)
                imag = F.grid_sample(inp.imag, grid_rep, mode="bilinear", align_corners=self.align_corners)
                samp = torch.complex(real, imag)  # (B*K,1,H_out,W_out)
            else:
                samp = F.grid_sample(inp, grid_rep, mode="bilinear", align_corners=self.align_corners)

            samp = samp.reshape(B, K, 1, H_out, W_out)             # (B,K,1,H,W)

            out = (w * samp).sum(dim=1)                             # (B,1,H,W) true complex mix
            outputs[kk] = out

        return outputs
