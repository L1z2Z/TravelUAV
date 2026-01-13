import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple
from .memory_utils import is_close, distance
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
    # current_anchor is a per-batch list of anchors (len == B).
    current_anchor: Optional[list[Anchor]] = None
    # Memory tensors keep batch dimension (B, ...).
    M_wsem: Optional[torch.Tensor] = None
    M_wenv: Optional[torch.Tensor] = None
    M_detail: Optional[torch.Tensor] = None

    def prepare_M_env(self):
        pass
    def prepare_M_detail(self):
        pass

class WorkingMemoryGenerator(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_llm: int,
        n_sem: int = 2,
        n_env: int = 2,
        n_detail: int = 8,
        n_heads: int = 8,
        dropout: float = 0.0,
        ffn_mult: int = 2,
        use_ffn: bool = True,
    ):
        super(WorkingMemoryGenerator, self).__init__()
        # Define layers here
        self.d_in = d_in
        self.d_llm = d_llm
        self.n_sem = n_sem
        self.n_env = n_env
        self.n_detail = n_detail

        self.proj = nn.Linear(d_in, d_llm)
        self.vision_ln = nn.LayerNorm(d_llm)

        self.sem_queries = nn.Parameter(torch.randn(n_sem, d_llm) * 0.02)
        self.env_queries = nn.Parameter(torch.randn(n_env, d_llm) * 0.02)
        self.detail_queries = nn.Parameter(torch.randn(n_detail, d_llm) * 0.02)

        self.sem_attention = nn.MultiheadAttention(d_llm, n_heads, dropout=dropout, batch_first=True)
        self.vis_attention = nn.MultiheadAttention(d_llm, n_heads, dropout=dropout, batch_first=True)

        self.use_ffn = use_ffn
        if use_ffn:
            hidden = d_llm * ffn_mult
            self.shared_ffn = nn.Sequential(
                nn.LayerNorm(d_llm),
                nn.Linear(d_llm, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_llm),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        raw_image_features: torch.Tensor,
        semantic_feature: torch.Tensor,
        semantic_key_padding_mask: Optional[torch.Tensor] = None,
        anchor: Optional[list[Anchor]] = None,
    ) -> WorkingMemory:
        B, _, _ = raw_image_features.shape # B x P x D_in

        I = self.proj(raw_image_features)  # B x P x d_llm
        I = self.vision_ln(I)
        S = semantic_feature  # B x S_len x d_llm

        q_sem = self.sem_queries.unsqueeze(0).expand(B, -1, -1)  # B x n_sem x d_llm
        q_env = self.env_queries.unsqueeze(0).expand(B, -1, -1)  # B x n_env x d_llm
        q_detail = self.detail_queries.unsqueeze(0).expand(B, -1, -1)  # B x n_detail x d_llm

        sem, _ = self.sem_attention(
            q_sem,
            S,
            S,
            key_padding_mask=semantic_key_padding_mask,
            need_weights=False,
        )  # B x n_sem x d_llm

        q_vis = torch.cat([q_env, q_detail], dim=1)  # B x (n_env+n_detail) x d_llm
        vis, _ = self.vis_attention(q_vis, I, I, need_weights=False)  # B x (n_env+n_detail) x d_llm
        env, details = torch.split(vis, [self.n_env, self.n_detail], dim=1)

        if self.use_ffn:
            sem = sem + self.shared_ffn(sem)  # B x n_sem x d_llm
            env = env + self.shared_ffn(env)  # B x n_env x d_llm
            details = details + self.shared_ffn(details)  # B x n_detail x d_llm

        workingmemory = WorkingMemory(
            current_anchor=anchor,
            M_wsem=sem,
            M_wenv=env,
            M_detail=details
        )

        return workingmemory



'''
Task Memory Part
'''
@dataclass
class TaskMemoryNode:
    node_id: int
    visit_history: list[int] = field(default_factory=list)  # 记录该节点为导航中第几个抵达的节点，可能重复抵达
    anchors: list[Anchor] = field(default_factory=list)     # 时间步和位置信息
    M_sem: list[Any] = field(default_factory=list)
    M_env: list[Any] = field(default_factory=list)
    M_act: list[Any] = field(default_factory=list)
    score: float = 0.0
    #orientation_change: float                               # 航向改变角度
    #types: list[bool] = field(default_factory=list)         # True表示landmark；False表示航向大幅改变
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
    node_ids:Tuple[int, int]
    distance:float = 0.0
    orientation:float = 0.0

@dataclass
class TaskMemoryGraph:
    nodes:list[TaskMemoryNode] = field(default_factory=list)            # 存储所有节点
    temporary_nodes:list[TaskMemoryNode] = field(default_factory=list)  # 存储临时节点，待合并和删除
    nodes_sequence:list[int] = field(default_factory=list)              # 按抵达的时间顺序存储节点id，可以重复（重复抵达）
    neighbors:list[set[int]] = field(default_factory=list)
    edges:dict[tuple[int, int], TaskMemoryEdge] = field(default_factory=dict)
    max_nodes:int = TASK_MEMORY_MAX_NODES

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def _pool_env(self, m_env: Any) -> torch.Tensor:
        """Convert stored env memory to a [H] or [B,H] tensor for aggregation."""
        if isinstance(m_env, list) and len(m_env) > 0:
            m_env = m_env[-1]
        if not isinstance(m_env, torch.Tensor):
            raise TypeError(f"M_env must be torch.Tensor, got {type(m_env)}")

        # Common shapes:
        # - [B, n_env, H] from working memory generator
        # - [B, H]
        # - [H]
        if m_env.dim() == 3:
            return m_env.mean(dim=1)  # [B, H]
        if m_env.dim() == 2:
            return m_env  # [B, H]
        if m_env.dim() == 1:
            return m_env.unsqueeze(0)  # [1, H]
        raise ValueError(f"Unsupported M_env shape: {tuple(m_env.shape)}")

    def add_temporary_node(self, anchor:Anchor, m_sem:Any, m_env:Any, m_act:Any, type:bool=True):
        node_id = len(self.temporary_nodes)
        self.temporary_nodes.append(
            TaskMemoryNode(
                node_id=node_id,
                visit_history=[],
                anchors=[anchor],
                M_sem=[m_sem],
                M_env=[m_env],
                M_act=[m_act],
            )
        )
        
    
    def add_edge(self):
        if self.num_nodes < 2:
            return
        a = self.nodes[-2].node_id
        b = self.nodes[-1].node_id
        edge = TaskMemoryEdge(
            node_ids=(a, b),
            distance=distance(self.nodes[-2].anchors, self.nodes[-1].anchors),
        )
        self.edges[(a, b)] = edge
        assert len(self.neighbors) > max(a, b), (
            f"neighbors not initialized for node ids: a={a}, b={b}, len(neighbors)={len(self.neighbors)}. "
            "Expected invariant: len(neighbors) == len(nodes) and node_id matches index."
        )
        self.neighbors[a].add(b)
        self.neighbors[b].add(a)

    def prune(self, window_length:int):
        # 将temporary_nodes合并为一个new_node
        self.temporary_nodes = self.temporary_nodes[ -window_length : ]  # 仅保留最近window_length个临时节点
        max_score_node = max(self.temporary_nodes, key=lambda x: x.score)
        topk_score_nodes = sorted(self.temporary_nodes, key=lambda x: x.score, reverse=True)[:3]

        # Aggregate env features across top-k temporary nodes.
        pooled_envs = [self._pool_env(n.M_env) for n in topk_score_nodes]
        env_mean = torch.stack(pooled_envs, dim=0).mean(dim=0)  # [B, H]

        new_node = TaskMemoryNode(
            node_id = len(self.nodes),
            visit_history = [len(self.nodes_sequence)],
            anchors = [max_score_node.anchors[-1]],
            M_sem = [max_score_node.M_sem[-1]],
            M_env = [env_mean],
            M_act = [max_score_node.M_act[-1]],
            score = max_score_node.score,
        )

        # 检查new_node是否是旧节点
        for node in self.nodes:
            if len(new_node.anchors) > 0 and is_close(new_node.anchors[-1], node.anchors):
                # 若为旧节点，则更新信息
                node.visit_history.append(len(self.nodes_sequence))
                node.anchors.append(new_node.anchors[-1])
                node.M_sem.append(new_node.M_sem[-1])
                node.M_env.append(new_node.M_env[-1])
                node.M_act.append(new_node.M_act[-1])

                self.add_edge()
                self.nodes_sequence.append(node.node_id)
                self.temporary_nodes = []               # 清空临时节点列表
                return
            else:
                continue

        # 若为新节点，则添加到nodes中
        self.nodes.append(new_node)
        self.nodes_sequence.append(new_node.node_id)
        self.neighbors.append(set())
        self.add_edge()
        self.temporary_nodes = []               # 清空临时节点列表

    def neighbors_of(self, node_id:int) -> list[int]:
        return self.neighbors[node_id]
    
    def retrieve(self, query:Any, topk:int=TASK_MEMORY_DEFAULT_TOPK) -> list[TaskMemoryNode]:
        pass

'''
Open Memory Part
'''
@dataclass
class OpenMemoryBuffer:
    anchors:list[Anchor] = field(default_factory=list)
    M_env:list[Any] = field(default_factory=list)
    M_detail:list[Any] = field(default_factory=list)
    max_size:int = OPEN_MEMORY_MAX_SIZE

    def add_memory(self, anchor:Anchor, M_env:Any, M_detail:Any):
        self.anchors.append(anchor)
        self.M_env.append(M_env)
        self.M_detail.append(M_detail)

    def retrieve(self, query:torch.Tensor, topk:int=OPEN_MEMORY_DEFAULT_TOPK) -> list[Any]:
        if len(self.anchors) == 0:
            return []

        def _pool_to_vec(x: Any) -> torch.Tensor:
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(x)}")
            # Accept common shapes produced by working memory generator.
            # - [1, n_env, H] -> [H]
            # - [B, n_env, H] -> [H] (mean over B)
            # - [1, H] -> [H]
            # - [H] -> [H]
            if x.dim() == 3:
                x = x.mean(dim=1)  # [B, H]
            if x.dim() == 2:
                x = x.mean(dim=0)  # [H]
            if x.dim() == 1:
                return x
            raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")

        q = _pool_to_vec(query)
        cos_similarities = [float(F.cosine_similarity(q, _pool_to_vec(m), dim=0).item()) for m in self.M_env]
        topk = min(int(topk), len(cos_similarities))
        sorted_indices = sorted(range(len(cos_similarities)), key=lambda i: cos_similarities[i], reverse=True)
        topk_indices = sorted_indices[:topk]
        return [self.M_env[i] for i in topk_indices]
