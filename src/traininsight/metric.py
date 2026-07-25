from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
from .training_state import TrainingState

KINDS = ("training", "capability")


class Metric(ABC):
    """General interface: subclass, set name/kind, implement measure(state)."""

    name = "metric"
    kind = "training"

    def __init__(self, name: Optional[str] = None, kind: Optional[str] = None):
        if name:
            self.name = name
        if kind:
            self.kind = kind
        if not self.name or self.kind not in KINDS:
            raise ValueError(f"need name and kind in {KINDS}")

    @abstractmethod
    def measure(self, state: TrainingState) -> Any:
        raise NotImplementedError

    def __call__(self, s: TrainingState) -> Any:
        return self.measure(s)


class TrainingMetric(Metric):
    kind = "training"


class CapabilityMetric(Metric):
    kind = "capability"
