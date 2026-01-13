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
        # Per-batch storages. Each batch element corresponds to an independent task.
        self.task_graph: list[TaskMemoryGraph] = []
        self.open_buffer: list[OpenMemoryBuffer] = []
        self.working_memory_generator = None
        self.instruction_embeds = None
        self.instruction_attention_mask = None

        self._route_step: int = 0
        # Per-batch task-window state.
        self._in_task_window: list[bool] = []
        self._task_consecutive: list[int] = []
        self._open_consecutive_in_window: list[int] = []
        self._task_window_start_step: list[Optional[int]] = []

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
        wm = self.working_memory

        anchors = self.working_memory.current_anchor
        if anchors is None:
            self._route_step += 1
            return None

        if isinstance(anchors, (list, tuple)):
            batch_size = len(anchors)
        else:
            # Fallback: infer batch size from tensors if possible.
            if isinstance(wm.M_wenv, torch.Tensor) and wm.M_wenv.dim() >= 1:
                batch_size = int(wm.M_wenv.shape[0])
            else:
                batch_size = 1
            anchors = [anchors]

        # Ensure per-batch storages/state match current batch size.
        if len(self.task_graph) != batch_size:
            self.task_graph = [TaskMemoryGraph() for _ in range(batch_size)]
        if len(self.open_buffer) != batch_size:
            self.open_buffer = [OpenMemoryBuffer() for _ in range(batch_size)]
        if len(self._in_task_window) != batch_size:
            self._in_task_window = [False for _ in range(batch_size)]
            self._task_consecutive = [0 for _ in range(batch_size)]
            self._open_consecutive_in_window = [0 for _ in range(batch_size)]
            self._task_window_start_step = [None for _ in range(batch_size)]

        # 标记时间戳：必须修改 self.working_memory.current_anchor（后面还会用到）
        if isinstance(self.working_memory.current_anchor, (list, tuple)):
            for a in self.working_memory.current_anchor:
                if a is not None:
                    a.t = self._route_step
        else:
            if self.working_memory.current_anchor is not None:
                self.working_memory.current_anchor.t = self._route_step
        
        # 任务关联度判断
        # 分数越高，表示与当前任务越相关
        # Score is mapped to [0, 1] for thresholding.
        import pdb; pdb.set_trace()
        # 检查一下instr和env是不是想要的

        # ---------- PDB manual debug (copy/paste into pdb) ----------
        # 1) Print prompts as raw text
        # print("=== prompts (raw) ===")
        # if prompts is None:
        #     print(None)
        # else:
        #     print("\n---\n".join(prompts))
        #
        # 2) Tokenize prompts with the tokenizer used for instruction_input_ids
        #    (This creates check_instruction_ids for manual comparison)
        # from transformers import AutoTokenizer
        # tok_path = getattr(self.config, "_name_or_path", None) or getattr(self.config, "name_or_path", None)
        # tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=False)
        # if tokenizer.pad_token is None:
        #     tokenizer.pad_token = tokenizer.eos_token
        # check_instruction_ids = tokenizer(
        #     prompts,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True,
        # ).input_ids
        # print("=== check_instruction_ids ===")
        # print(check_instruction_ids)
        #
        # 3) Print instruction_input_ids (provided by wrapper)
        # print("=== instruction_input_ids ===")
        # print(instruction_input_ids)
        # -----------------------------------------------------------

        # score_vec: [B] in [0, 1]
        if self.instruction_embeds is not None and wm.M_wenv is not None:
            instr = masked_mean_pool(
                self.instruction_embeds,
                self.instruction_attention_mask,
                dim=1,
            )  # [B, H]
            env = masked_mean_pool(wm.M_wenv, None, dim=1)  # [B, H]

            cos = F.cosine_similarity(instr, env, dim=-1)  # [B]
            score_vec = ((cos + 1.0) * 0.5).clamp(0.0, 1.0)
        else:
            score_vec = torch.zeros(batch_size, device=wm.M_wenv.device if isinstance(wm.M_wenv, torch.Tensor) else None)

        frame_step = int(self._route_step)

        for i in range(batch_size):
            # 若没有任务记忆，则强制路由到任务记忆
            if int(getattr(self.task_graph[i], "num_nodes", 0)) == 0:
                route = "task"
                reason = "start_node"
            else:
                score_i = float(score_vec[i].item())
                route = "task" if score_i >= float(MEMORY_MANAGER_ROUTE_THRESHOLD) else "open"
                reason = "relatedness_threshold"

            anchor_i = anchors[i]
            m_sem_i = wm.M_wsem[i : i + 1] if isinstance(wm.M_wsem, torch.Tensor) else wm.M_wsem
            m_env_i = wm.M_wenv[i : i + 1] if isinstance(wm.M_wenv, torch.Tensor) else wm.M_wenv
            m_detail_i = wm.M_detail[i : i + 1] if isinstance(wm.M_detail, torch.Tensor) else wm.M_detail

            if route == "task":
                self.task_graph[i].add_temporary_node(
                    anchor=anchor_i,
                    m_sem=m_sem_i,
                    m_env=m_env_i,
                    m_act=None,
                    type=True,  # True表示landmark节点
                )
            else:
                self.open_buffer[i].add_memory(
                    anchor_i,
                    m_env_i,
                    m_detail_i,
                )

            # ---- Task window 用于任务记忆的剪枝（per-batch）----
            # 连续TASK_WINDOW_ENTER个帧路由到任务记忆，则进入任务窗口
            # 连续TASK_WINDOW_EXIT个帧路由到开放记忆，则退出任务窗口
            if route == "task":
                self._open_consecutive_in_window[i] = 0
                self._task_consecutive[i] += 1

                if (not self._in_task_window[i]) and self._task_consecutive[i] >= TASK_WINDOW_ENTER:  # 进入任务窗口
                    self._in_task_window[i] = True
                    # window starts at the first frame of this consecutive-run
                    self._task_window_start_step[i] = frame_step - TASK_WINDOW_ENTER + 1
            else:  # route == "open"
                if self._in_task_window[i]:
                    # Allow short occlusion/noise: do NOT reset _task_consecutive until we confirm exit.
                    self._open_consecutive_in_window[i] += 1
                    if self._open_consecutive_in_window[i] >= TASK_WINDOW_EXIT:  # 退出任务窗口
                        window_start = self._task_window_start_step[i]
                        window_end = frame_step - TASK_WINDOW_EXIT
                        if window_start is not None and window_end >= window_start:
                            self.task_graph[i].prune(window_length=window_end - window_start + 1)

                        self._in_task_window[i] = False
                        self._open_consecutive_in_window[i] = 0
                        self._task_window_start_step[i] = None
                        self._task_consecutive[i] = 0
                else:
                    # Not in task window: open breaks the "N consecutive task" condition.
                    self._task_consecutive[i] = 0

        self._route_step += 1

        return None

    def retrieve_memory(self, anchor: Anchor, top_k: int) -> list[Any]:
        pass