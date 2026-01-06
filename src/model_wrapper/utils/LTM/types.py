import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Optional
from utils.LTM.utils import is_close, distance
from .hyper_parameter import (
    OPEN_MEMORY_DEFAULT_TOPK,
    OPEN_MEMORY_MAX_SIZE,
    TASK_MEMORY_DEFAULT_TOPK,
    TASK_MEMORY_MAX_NODES,
)


@dataclass
class Anchor:
    t:float
    x:float
    y:float
    z:float
    yaw:float = 0.0
    frame_id:int = 0

    def location_as_tensor(self):
        return torch.tensor([self.x, self.y, self.z])

@dataclass
class TriggerInfo:
    caption:list[str] = field(default_factory=list)
    sign:dict[Any, Any] = field(default_factory=dict)

    def add_reason(self, new_reason:str):
        self.reason.append(new_reason)

'''
Working Memory Part
'''
@dataclass
class WorkingMemory:
    current_anchor: Anchor
    M_wsem: Any = None
    M_wenv: Any = None
    M_detail: Any = None

    def prepare_M_env(self):
        pass
    def prepare_M_detail(self):
        pass

@dataclass
class WorkingMemoryGenerator(nn.Module):
    def __init__(self, d_in:int, d_llm:int, n_place:int = 2, n_detail:int = 8, n_heads:int = 8, dropout:float = 0.0):
        super(WorkingMemoryGenerator, self).__init__()
        # Define layers here
        self.d_in = d_in
        self.d_llm = d_llm
        self.n_place = n_place
        self.n_detail = n_detail

        self.proj = nn.Linear(d_in, d_llm)
        self.vision_ln = nn.LayerNorm(d_llm)

        self.place_queries = nn.Parameter(torch.randn(n_place, d_llm) * 0.02)
        self.detail_queries = nn.Parameter(torch.randn(n_detail, d_llm) * 0.02)

        self.place_attention = nn.MultiheadAttention(d_llm, n_heads, dropout=dropout, batch_first=True)
        self.detail_attention = nn.MultiheadAttention(d_llm, n_heads, dropout=dropout, batch_first=True)

        self.place_ffn = nn.Sequential(
            nn.LayerNorm(d_llm),
            nn.Linear(d_llm, d_llm * 4),
            nn.GELU(),
            nn.Linear(d_llm * 4, d_llm),
            nn.Dropout(dropout),
        )

        self.detail_ffn = nn.Sequential(
            nn.LayerNorm(d_llm),
            nn.Linear(d_llm, d_llm * 4),
            nn.GELU(),
            nn.Linear(d_llm * 4, d_llm),
            nn.Dropout(dropout),
        )

    def forward(self, raw_image_features: torch.Tensor) -> WorkingMemory:
        # Implement the forward pass to generate working memory
        B, _, _ = raw_image_features.shape # B x P x D_in

        x = self.proj(raw_image_features)  # B x P x d_llm
        x = self.vision_ln(x)

        q_place = self.place_queries.unsqueeze(0).expand(B, -1, -1)  # B x n_place x d_llm
        q_detail = self.detail_queries.unsqueeze(0).expand(B, -1, -1)  # B x n_detail x d_llm

        place, _ = self.place_attention(q_place, x, x, need_weights=False)  # B x n_place x d_llm
        details, _ = self.detail_attention(q_detail, x, x, need_weights=False)  # B x n_detail x d_llm

        place = place + self.place_ffn(place)  # B x n_place x d_llm
        details = details + self.detail_ffn(details)  # B x n_detail x d_llm

        workingmemory = WorkingMemory(
            current_anchor=None,
            M_wsem=None,
            M_wenv=place,
            M_detail=details
        )

        return workingmemory


'''
Task Memory Part
'''
@dataclass
class TaskMemoryNode:
    node_ids: list[int] = field(default_factory=list) # 按时间顺序记录该节点对应的所有node_id
    anchors: list[Anchor] = field(default_factory=list)
    M_sem: list[Any] = field(default_factory=list)
    M_env: list[Any] = field(default_factory=list)
    M_act: list[Any] = field(default_factory=list)
    types: list[bool] = field(default_factory=list) # True表示landmark；False表示航向大幅改变
    #trigger_score: float = 0.0
    #caption:list[str] = field(default_factory=list)
    #sign:dict[Any, Any] = field(default_factory=dict) # 图片、动作等的摘要信息
    
    def as_kv(self):
        pass

    def as_embedding(self):
        pass

    def captioning(self):
        return self.caption


@dataclass
class TaskMemoryEdge:
    start_id:int
    end_id:int
    distance:float = 0.0

@dataclass
class TaskMemoryGraph:
    nodes:list[TaskMemoryNode] = field(default_factory=list)
    neighbors:list[list[int]] = field(default_factory=list)
    edges:dict[tuple[int, int], TaskMemoryEdge] = field(default_factory=dict)
    num_nodes:int = 0
    max_nodes:int = TASK_MEMORY_MAX_NODES

    def add_node(self, anchor:Anchor, m_sem:Any, m_env:Any, m_act:Any, type:bool=False):
        # 先判断是否为新节点
        for node in self.nodes:
            if is_close(anchor, node.anchors):
                # 若为旧节点，则更新信息
                node.node_ids.append(self.num_nodes)
                node.anchors.append(anchor)
                node.M_sem.append(m_sem)
                node.M_env.append(m_env)
                node.M_act.append(m_act)
                node.types.append(type)
                break
            else:
                # 若为新节点，则创建并添加        
                self.nodes.append(
                    TaskMemoryNode(
                        node_ids=[self.num_nodes],
                        anchors=[anchor],
                        M_sem=[m_sem],
                        M_env=[m_env],
                        M_act=[m_act],
                        types=[type]
                    )
                )
                break
        self.num_nodes += 1
    
    def add_edge(self):
        self.edges.append(
            TaskMemoryEdge(
                start_id = self.num_nodes - 2,
                end_id = self.num_nodes - 1,
                distance = distance(self.nodes[-2].anchors, self.nodes[-1].anchors)
            )
        )
        self.neighbors[self.num_nodes - 2].append(self.num_nodes - 1)
        self.neighbors[self.num_nodes - 1].append(self.num_nodes - 2)

    def neighbors_of(self, node_id:int) -> list[int]:
        return self.neighbors[node_id]

    def prune(self, strategy:str='closest'):
        pass

    def retrieve(self, query:Any, topk:int=4) -> list[TaskMemoryNode]:
        pass

'''
Open Memory Part
'''
@dataclass
class OpenMemoryList:
    anchors:list[Anchor] = field(default_factory=list)
    M_open:list[Any] = field(default_factory=list)
    max_size:int = OPEN_MEMORY_MAX_SIZE

    def add_memory(self, anchor:Anchor, m_open:Any):
        self.anchors.append(anchor)
        self.M_open.append(m_open)

    def retrieve(self, query:torch.Tensor, topk:int=OPEN_MEMORY_DEFAULT_TOPK) -> list[Any]:
        if len(self.anchors) == 0:
            return []
        
        cos_similarities = [F.cosine_similarity(query, m) for m in self.M_open]
        sorted_indices = sorted(range(len(cos_similarities)), key=lambda i: cos_similarities[i], reverse=True)
        topk_indices = sorted_indices[:topk]
        return [self.M_open[i] for i in topk_indices]
