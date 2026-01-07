import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Optional
from .memory_utils import is_close, distance
from .types import Anchor, TriggerInfo, WorkingMemory,WorkingMemoryGenerator, TaskMemoryGraph, TaskMemoryEdge, TaskMemoryNode, OpenMemoryBuffer

class MemoryManager(nn.Module):
    def __init__(self):
        super(MemoryManager, self).__init__()
        self.working_memory = WorkingMemory(
            current_anchor=None,
            M_wsem=None,
            M_wenv=None,
            M_detail=None
        )
        self.task_graph = None
        self.open_buffer = None
        self.working_memory_generator = None

    def _ensure_working_memory_generator(self, raw_image_features: torch.Tensor, d_llm: int):
        if self.working_memory_generator is not None:
            return

        self.working_memory_generator = WorkingMemoryGenerator(
            d_in=int(raw_image_features.shape[-1]),
            d_llm=d_llm,
            n_sem=2,
        ).to(device=raw_image_features.device, dtype=raw_image_features.dtype)

    def update_working_memory(
        self,
        raw_image_features: torch.Tensor,
        semantic_memory: torch.Tensor,
        d_llm: int,
        semantic_key_padding_mask: Optional[torch.Tensor] = None,
        anchor: Anchor = None,
    ):
        self._ensure_working_memory_generator(raw_image_features=raw_image_features, d_llm=d_llm)

        if semantic_key_padding_mask is not None:
            semantic_key_padding_mask = semantic_key_padding_mask.to(device=raw_image_features.device)

        self.working_memory = self.working_memory_generator(
            raw_image_features=raw_image_features,
            semantic_feature=semantic_memory,
            semantic_key_padding_mask=semantic_key_padding_mask,
            anchor=anchor,
        )

    def add_task_memory_node(self, node: TaskMemoryNode):
        pass

    def add_open_memory(self, anchor: Anchor, data: Any):
        pass

    def retrieve_memory(self, anchor: Anchor, top_k: int) -> list[Any]:
        pass