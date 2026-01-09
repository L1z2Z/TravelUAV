import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Optional
import os
from .memory_utils import masked_mean_pool
from .types import Anchor, TriggerInfo, WorkingMemory, WorkingMemoryGenerator, TaskMemoryGraph, TaskMemoryEdge, OpenMemoryBuffer
from .hyper_parameter import MEMORY_MANAGER_ROUTE_THRESHOLD , TASK_WINDOW_ENTER , TASK_WINDOW_EXIT


class MemoryManager(nn.Module):
    def __init__(self):
        super(MemoryManager, self).__init__()
        self.working_memory = WorkingMemory(
            current_anchor=None,
            M_wsem=None,
            M_wenv=None,
            M_detail=None
        )
        self.task_graph = TaskMemoryGraph()
        self.open_buffer = OpenMemoryBuffer()
        self.working_memory_generator = None
        self.instruction_embeds = None
        self.instruction_attention_mask = None

        self._route_step: int = 0
        self._in_task_window: bool = False
        self._task_consecutive: int = 0
        self._open_consecutive_in_window: int = 0
        self._task_window_start_step: Optional[int] = None

    def update_instruction_embedding(
        self,
        instruction_embeds: torch.Tensor,
        instruction_attention_mask: Optional[torch.Tensor] = None,
    ):
        self.instruction_embeds = instruction_embeds
        self.instruction_attention_mask = instruction_attention_mask

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

    def memory_storge(self):
        self.route_working_memory()

    def route_working_memory(self):
        """Route current working memory into task memory or open memory.

        This function updates internal storages (task graph / open buffer) and
        task-window state. It does not return any value.
        """
        
        self.working_memory.current_anchor.t = self._route_step     # 标记时间戳
        wm = self.working_memory
        
        # 任务关联度判断
        # 分数越高，表示与当前任务越相关
        # Score is mapped to [0, 1] for thresholding.
        import pdb; pdb.set_trace()
        # 检查一下instr和env是不是想要的
        score = 0.0
        if self.instruction_embeds is not None and wm.M_wenv is not None:
            instr = masked_mean_pool(
                self.instruction_embeds,
                self.instruction_attention_mask,
                dim=1,
            )  # [B, H]
            env = masked_mean_pool(wm.M_wenv, None, dim=1)  # [B, H]

            cos = F.cosine_similarity(instr, env, dim=-1)
            score = float(((cos.mean() + 1.0) * 0.5).clamp(0.0, 1.0).item())

        # 若没有任务记忆，则强制路由到任务记忆
        if int(getattr(self.task_graph, "num_nodes", 0)) == 0:
            route = "task"
            reason = "start_node"
        else:
            route = "task" if score >= float(MEMORY_MANAGER_ROUTE_THRESHOLD) else "open"
            reason = "relatedness_threshold"

        if route == "task":
            self.task_graph.add_temporary_node(
                anchor=wm.current_anchor,
                m_sem=wm.M_wsem,
                m_env=wm.M_wenv,
                m_act=None,
                type=True,              # True表示landmark节点
            )
        else:
            self.open_buffer.add_memory(
                wm.current_anchor,
                wm.M_wenv,
                wm.M_detail,
            )

        # ---- Task window 用于任务记忆的剪枝 ----
        # 连续TASK_WINDOW_ENTER个帧路由到任务记忆，则进入任务窗口
        # 连续TASK_WINDOW_EXIT个帧路由到开放记忆，则退出任务窗口
        frame_step = int(self._route_step)
        if route == "task":
            self._open_consecutive_in_window = 0
            self._task_consecutive += 1

            if (not self._in_task_window) and self._task_consecutive >= TASK_WINDOW_ENTER:  # 进入任务窗口
                self._in_task_window = True
                # window starts at the first frame of this consecutive-run
                self._task_window_start_step = frame_step - TASK_WINDOW_ENTER + 1
        else:  # route == "open"
            if self._in_task_window:
                # Allow short occlusion/noise: do NOT reset _task_consecutive until we confirm exit.
                self._open_consecutive_in_window += 1
                if self._open_consecutive_in_window >= TASK_WINDOW_EXIT:    # 退出任务窗口
                    window_start = self._task_window_start_step
                    window_end = frame_step - TASK_WINDOW_EXIT
                    if window_start is not None and window_end >= window_start:
                        self.task_graph.prune(window_length=window_end - window_start + 1)

                    self._in_task_window = False
                    self._open_consecutive_in_window = 0
                    self._task_window_start_step = None
                    self._task_consecutive = 0
            else:
                # Not in task window: open breaks the "N consecutive task" condition.
                self._task_consecutive = 0

        self._route_step += 1

        return None

    def retrieve_memory(self, anchor: Anchor, top_k: int) -> list[Any]:
        pass