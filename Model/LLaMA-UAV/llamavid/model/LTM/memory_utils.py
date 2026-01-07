from __future__ import annotations

import torch

from typing import TYPE_CHECKING

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


