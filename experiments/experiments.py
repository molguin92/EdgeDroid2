from typing import Any, Callable, Dict, NamedTuple

import numpy as np

from edgedroid.models import *


class ExperimentConfig(NamedTuple):
    timing_model: ExecutionTimeModel
    sampling_scheme: BaseSamplingPolicy
    metadata: Dict[str, Any] = {}


experiments: Dict[str, Callable[[], ExperimentConfig]] = {
    "legacy": lambda: ExperimentConfig(
        timing_model=LegacyModel(),
        sampling_scheme=LegacySamplingPolicy.from_default_data(),
        metadata={
            "timing_model": "legacy",
            "sampling_policy": "legacy",
        },
    ),
    "curve-low-neuro": lambda: ExperimentConfig(
        timing_model=MultiCurveFittingExecutionTimeModel(neuroticism=0.0),
        sampling_scheme=ZeroWaitSamplingPolicy.from_default_data(),
        metadata={
            "timing_model": "curve-low",
            "sampling_policy": "zero-wait",
        },
    ),
    "curve-high-neuro": lambda: ExperimentConfig(
        timing_model=MultiCurveFittingExecutionTimeModel(neuroticism=1.0),
        sampling_scheme=ZeroWaitSamplingPolicy.from_default_data(),
        metadata={
            "timing_model": "curve-high",
            "sampling_policy": "zero-wait",
        },
    ),
    "first-order": lambda: ExperimentConfig(
        timing_model=FirstOrderETM(),
        sampling_scheme=ZeroWaitSamplingPolicy.from_default_data(),
        metadata={
            "timing_model": "first-order",
            "sampling_policy": "zero-wait",
        },
    ),
    "first-order-median": lambda: ExperimentConfig(
        timing_model=FirstOrderAggregateETM(aggregate_fn=np.median),
        sampling_scheme=ZeroWaitSamplingPolicy.from_default_data(),
        metadata={
            "timing_model": "first-order-median",
            "sampling_policy": "zero-wait",
        },
    ),
}
__all__ = ["experiments"]
