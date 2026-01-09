"""Centralized hyper-parameters for the LTM module.

Goal
----
Keep all tunable knobs for LTM (Long-Term Memory) in one place, so other
modules (e.g., `types.py`) can import defaults without hard-coding numbers.

How to extend
-------------
1) Add a field to one of the dataclasses below (TaskMemoryHP/OpenMemoryHP).
2) Use it from other files via `LTM_HP.<section>.<field>` or via the exported
   constants (e.g., TASK_MEMORY_MAX_NODES).
3) (Optional) override values at runtime via `load_ltm_hp_from_env()`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from typing import Any, Dict


def _env_int(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None or value == "":
		return default
	try:
		return int(value)
	except ValueError as exc:
		raise ValueError(f"Env var {name} must be int, got: {value!r}") from exc

def _env_float(name: str, default: float) -> float:
	value = os.getenv(name)
	if value is None or value == "":
		return default
	try:
		return float(value)
	except ValueError as exc:
		raise ValueError(f"Env var {name} must be float, got: {value!r}") from exc

@dataclass(frozen=True)
class TaskMemoryHP:
	"""Hyper-parameters for Task Memory (graph-based)."""

	max_nodes: int = 16
	default_topk: int = 4

	# Reserved knobs you may want later:
	# prune_strategy: str = "closest"
	# merge_distance_threshold: float = 1.0

@dataclass(frozen=True)
class OpenMemoryHP:
	"""Hyper-parameters for Open Memory (list-based)."""

	max_size: int = 64
	default_topk: int = 4
	sampling_ratio: float = 10 # 间隔多少帧采样一次

@dataclass(frozen=True)
class MemoryMenagerHP:
	"""Hyper-parameters for Memory Manager."""

	route_threshold: float = 1.0  # 工作记忆节点路由阈值
	task_window_enter: int = 3  # 进入任务窗口所需连续任务路由数
	task_window_exit: int = 3  # 退出任务窗口所需连续开放路由数


@dataclass(frozen=True)
class LTMHyperParameters:
	"""Top-level container for all LTM hyper-parameters."""

	task: TaskMemoryHP = field(default_factory=TaskMemoryHP)
	open: OpenMemoryHP = field(default_factory=OpenMemoryHP)
	memory_manager: MemoryMenagerHP = field(default_factory=MemoryMenagerHP)

	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)


def load_ltm_hp_from_env(base: LTMHyperParameters | None = None) -> LTMHyperParameters:
	"""Return a new config with optional overrides from environment variables.

	Supported env vars (optional):
	  - LTM_TASK_MAX_NODES
	  - LTM_TASK_DEFAULT_TOPK
	  - LTM_OPEN_MAX_SIZE
	  - LTM_OPEN_DEFAULT_TOPK
	  - LTM_MEMORY_MANAGER_ROUTE_THRESHOLD
	  - LTM_TASK_WINDOW_ENTER
	  - LTM_TASK_WINDOW_EXIT
	"""

	base = base or LTMHyperParameters()
	return LTMHyperParameters(
		task=TaskMemoryHP(
			max_nodes=_env_int("LTM_TASK_MAX_NODES", base.task.max_nodes),
			default_topk=_env_int("LTM_TASK_DEFAULT_TOPK", base.task.default_topk),
		),
		open=OpenMemoryHP(
			max_size=_env_int("LTM_OPEN_MAX_SIZE", base.open.max_size),
			default_topk=_env_int("LTM_OPEN_DEFAULT_TOPK", base.open.default_topk),
		),
		memory_manager=MemoryMenagerHP(
			route_threshold=_env_float("LTM_MEMORY_MANAGER_ROUTE_THRESHOLD", float(base.memory_manager.route_threshold)),
			task_window_enter=_env_int("LTM_TASK_WINDOW_ENTER", base.memory_manager.task_window_enter),
			task_window_exit=_env_int("LTM_TASK_WINDOW_EXIT", base.memory_manager.task_window_exit),
		),
	)


# Default singleton used by other modules.
LTM_HP: LTMHyperParameters = load_ltm_hp_from_env()


# Convenience exports (stable names for imports).
TASK_MEMORY_MAX_NODES: int = LTM_HP.task.max_nodes
TASK_MEMORY_DEFAULT_TOPK: int = LTM_HP.task.default_topk

OPEN_MEMORY_MAX_SIZE: int = LTM_HP.open.max_size
OPEN_MEMORY_DEFAULT_TOPK: int = LTM_HP.open.default_topk

MEMORY_MANAGER_ROUTE_THRESHOLD: float = LTM_HP.memory_manager.route_threshold
TASK_WINDOW_ENTER: int = LTM_HP.memory_manager.task_window_enter
TASK_WINDOW_EXIT: int = LTM_HP.memory_manager.task_window_exit

