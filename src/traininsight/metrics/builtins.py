from __future__ import annotations
from ..metric import CapabilityMetric
from ..training_state import TrainingState


class EvalLossMetric(CapabilityMetric):
    """Capability metric: eval loss probe, consistent naming.

    Class name EvalLossMetric matches metric name eval_loss and extra key eval_loss.
    Reads from extra[extra_key] where extra_key defaults to eval_loss.
    """

    name = "eval_loss"

    def __init__(self, extra_key: str = "eval_loss", name: str | None = None):
        # BEFORE: extra_key is key in TrainingState extra, default eval_loss
        super().__init__(name=name or self.name)  # AFTER: name = eval_loss concrete, class name EvalLossMetric matches metric name eval_loss
        # BEFORE: store extra key for consistent reading
        self.extra_key = extra_key  # AFTER: extra_key = eval_loss concrete

    def measure(self, s: TrainingState) -> float | None:
        # BEFORE: s is TrainingState with extra containing eval_loss, e.g., extra={"eval_loss": 0.42} shape scalar
        v = s.get_extra(self.extra_key)  # AFTER: v = 0.42 concrete float retrieved from extra[extra_key] where extra_key = eval_loss
        try:
            # BEFORE: v may be None or non-float
            return float(v) if v is not None else None  # AFTER: float(v) shape () e.g., 0.42 concrete or None
        except Exception:
            # BEFORE: v not convertible to float
            return None  # AFTER: None


# Alias for backward compatibility during refactor, now consistent: LossMetric is same as EvalLossMetric, both read eval_loss
class LossMetric(EvalLossMetric):
    """Deprecated alias for EvalLossMetric, kept for compatibility but now consistent name eval_loss."""

    name = "eval_loss"
