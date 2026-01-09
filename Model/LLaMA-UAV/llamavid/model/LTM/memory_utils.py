from __future__ import annotations

import torch

from typing import TYPE_CHECKING , Optional

if TYPE_CHECKING:
    from .types import Anchor

def is_close(anchor1: Anchor, anchor2: list[Anchor]) -> bool:
    if len(anchor2) == 0:
        raise ValueError("anchor2 must be a non-empty list of Anchor")
    
    p1 = anchor1.location_as_tensor()
    p2 = torch.stack([a.location_as_tensor() for a in anchor2], dim=0).mean(dim=0)
    d = torch.norm(p1 - p2).item()
    return d < 10

def distance(anchor1: list[Anchor], anchor2: list[Anchor]) -> float:
    if len(anchor1) == 0 or len(anchor2) == 0:
        raise ValueError("anchor1 and anchor2 must be non-empty lists of Anchor")
    
    p1 = torch.stack([a.location_as_tensor() for a in anchor1], dim=0).mean(dim=0)
    p2 = torch.stack([a.location_as_tensor() for a in anchor2], dim=0).mean(dim=0)
    d = torch.norm(p1 - p2).item()
    return d

def masked_mean_pool(x:torch.Tensor, mask:Optional[torch.Tensor], dim:int=1, eps:float=1.0) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.to(dtype=x.dtype, device=x.device)

    # Common case in this repo: x=[B, L, H], mask=[B, L], dim=1
    if x.dim() == 3 and mask.dim() == 2 and dim == 1:
        mask_ = mask.unsqueeze(-1)  # [B, L, 1]
        summed = (x * mask_).sum(dim=dim)  # [B, H]
        # denom is "number of valid tokens"; for all-padding rows, define mean as 0.
        denom = mask_.sum(dim=dim).clamp(min=eps)  # [B, 1]
        return summed / denom
    raise NotImplementedError(
        f"masked_mean_pool only supports x=[B,L,H], mask=[B,L], dim=1; got x={tuple(x.shape)}, mask={None if mask is None else tuple(mask.shape)}, dim={dim}"
    )
    
