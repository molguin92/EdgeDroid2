from typing import Any, Callable, Dict, NamedTuple

from edgedroid.models import *


class ExperimentConfig(NamedTuple):
    timing_model: ExecutionTimeModel
    sampling_scheme: BaseSamplingPolicy
    metadata: Dict[str, Any] = {}


experiments: Dict[str, Callable[[], ExperimentConfig]] = {
    "legacy": lambda: ExperimentConfig(
        timing_model=LegacyETM(),
        sampling_scheme=LegacySamplingPolicy.from_default_data(),
        metadata={
            "timing_model": "legacy",
            "sampling_policy": "legacy",
        },
    ),
    "empirical-low-neuro": lambda: ExperimentConfig(
        timing_model=EmpiricalETM(neuroticism=0.0),
        sampling_scheme=ZeroWaitSamplingPolicy.from_default_data(),
        metadata={
            "timing_model": "empirical-low",
            "sampling_policy": "zero-wait",
        },
    ),
    "empirical-high-neuro": lambda: ExperimentConfig(
        timing_model=EmpiricalETM(neuroticism=1.0),
        sampling_scheme=ZeroWaitSamplingPolicy.from_default_data(),
        metadata={
            "timing_model": "empirical-high",
            "sampling_policy": "zero-wait",
        },
    ),
    "fitted-low-neuro": lambda: ExperimentConfig(
        timing_model=FittedETM(neuroticism=0.0),
        sampling_scheme=ZeroWaitSamplingPolicy.from_default_data(),
        metadata={
            "timing_model": "fitted-low",
            "sampling_policy": "zero-wait",
        },
    ),
    "fitted-high-neuro": lambda: ExperimentConfig(
        timing_model=FittedETM(neuroticism=1.0),
        sampling_scheme=ZeroWaitSamplingPolicy.from_default_data(),
        metadata={
            "timing_model": "fitted-high",
            "sampling_policy": "zero-wait",
        },
    ),
}
__all__ = ["experiments"]
