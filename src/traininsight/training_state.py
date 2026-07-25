from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrainingState:
    """Immutable snapshot: monotonic step + open extra bag.

    Loss moved into extra with better name `eval_loss` to keep state minimal
    and avoid top-level special cases.
    """

    step: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_extra(self, key: str, default: Any = None) -> Any:
        return self.extra.get(key, default)
