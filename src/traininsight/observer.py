from __future__ import annotations
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional
from .metric import Metric
from .training_state import TrainingState


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _pearson(xs, ys) -> Optional[float]:
    pairs = [(float(a), float(b)) for a, b in zip(xs, ys) if _finite(a) and _finite(b)]
    if len(pairs) < 2:
        return None
    xv, yv = zip(*pairs)
    mx = sum(xv) / len(xv)
    my = sum(yv) / len(yv)
    num = sum((x - mx) * (y - my) for x, y in zip(xv, yv))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xv))
    dy = math.sqrt(sum((y - my) ** 2 for y in yv))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


class Observer:
    """Minimal observer: single store + correlate directly, no get_series/get_steps.

    Correlate always requires a and b (str or Metric), no infer.
    Steps are assumed monotonic, so common steps preserve insertion order without sort.
    """

    def __init__(self, metrics: Optional[List[Metric]] = None):
        self.metrics = list(metrics or [])
        if len({m.name for m in self.metrics}) != len(self.metrics):
            raise ValueError("duplicate metric names")
        self.data: Dict[str, Dict[int, Any]] = defaultdict(dict)

    def update(self, s: TrainingState) -> None:
        for m in self.metrics:
            try:
                v = m.measure(s)
            except Exception:
                continue
            if v is None:
                continue
            self.data[m.name][s.step] = v

    def correlate(self, metric_a, metric_b) -> Optional[float]:
        """Correlate two metrics. a/b must be Metric objects, asserted.

        No inference, no sorted, monotonic steps assumed.
        """
        assert isinstance(metric_a, Metric), "metric_a must be Metric"
        assert isinstance(metric_b, Metric), "metric_b must be Metric"
        name_a = metric_a.name
        name_b = metric_b.name

        metric_a_series = self.data.get(name_a)
        metric_b_series = self.data.get(name_b)
        # requires exist, no default {}
        if metric_a_series is None or metric_b_series is None:
            return None
        if not metric_a_series or not metric_b_series:
            return None
        # monotonic, no sorted, better than dict_a/dict_b/common_steps
        shared_steps = [step for step in metric_a_series if step in metric_b_series]
        if len(shared_steps) < 2:
            return None
        values_a = [metric_a_series[step] for step in shared_steps]
        values_b = [metric_b_series[step] for step in shared_steps]
        return _pearson(values_a, values_b)
