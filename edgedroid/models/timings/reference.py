#  Copyright (c) 2022 Manuel Olguín Muñoz <molguin@kth.se>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

from typing import Dict, Any, Callable

import numpy as np
from numpy import typing as npt
from scipy import stats

from .base import TTimingModel, ExecutionTimeModel
from .realistic import EmpiricalETM, TTFWindowKernel


class LegacyETM(EmpiricalETM):
    """
    EdgeDroid 1.0 execution time model.
    """

    def __init__(
            self,
            kernel: TTFWindowKernel,
            seed: int = 4,
            ttf_levels: int = 7,
    ):
        super(LegacyETM, self).__init__(
            kernel=kernel,
            neuroticism=0.0,
            ttf_levels=ttf_levels,
        )
        # as long as the seed is the same will generate the same sequence of execution times
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)

    def advance(self: TTimingModel, ttf: float | int) -> TTimingModel:
        super().advance(
            ttf=0
        )  # EdgeDroid 1.0 used a trace recorded in optimal conditions
        return self

    def reset(self) -> None:
        self._window = np.zeros(self._window.size, dtype=float)
        self._rng = np.random.default_rng(seed=self._seed)
        self._steps = 0


class ConstantETM(ExecutionTimeModel):
    """
    Returns a constant execution time.
    """

    def __init__(self, execution_time_seconds: float):
        super(ConstantETM, self).__init__()
        self._exec_time = execution_time_seconds

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "execution_time_seconds": float(self._exec_time),
        }

    def advance(self: TTimingModel, ttf: float | int) -> TTimingModel:
        # no-op
        return self

    def get_execution_time(self) -> float:
        return self._exec_time

    def get_expected_execution_time(self) -> float:
        return self.get_execution_time()

    def get_mean_execution_time(self) -> float:
        return self.get_execution_time()

    def state_info(self) -> Dict[str, Any]:
        return {}

    def reset(self) -> None:
        # no-op
        pass

    def get_cdf_at_instant(self, instant: float):
        return float(instant > self._exec_time)


class FirstOrderETM(ExecutionTimeModel):
    """
    Returns execution times sampled from a simple distribution.
    """

    def __init__(self):
        super(FirstOrderETM, self).__init__()
        data, *_ = self.get_data()
        self._exec_times = data["exec_time"].to_numpy()
        self._rng = np.random.default_rng()

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "execution_time_seconds": {
                "mean": float(self._exec_times.mean()),
                "std": float(self._exec_times.std()),
            }
        }

    def advance(self: TTimingModel, ttf: float | int) -> TTimingModel:
        # no-op
        return self

    def get_execution_time(self) -> float:
        return self._rng.choice(self._exec_times)

    def get_expected_execution_time(self) -> float:
        return self._exec_times.mean()

    def get_mean_execution_time(self) -> float:
        return self.get_expected_execution_time()

    def state_info(self) -> Dict[str, Any]:
        return {}

    def reset(self) -> None:
        # no-op
        pass

    def get_cdf_at_instant(self, instant: float) -> float:
        return self._exec_times[self._exec_times < instant].size / self._exec_times.size


class FirstOrderFittedETM(FirstOrderETM):
    def __init__(
            self,
            dist: stats.rv_continuous = stats.exponnorm,
    ):
        super(FirstOrderFittedETM, self).__init__()

        *self._dist_args, self._loc, self._scale = dist.fit(self._exec_times)
        self._dist: stats.rv_continuous = dist.freeze(
            loc=self._loc, scale=self._scale, *self._dist_args
        )
        self._dist.random_state = self._rng

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "execution_time_seconds": {
                "distribution": self._dist.__class__.__name__,
                "loc": self._loc,
                "scale": self._scale,
                "other": list(self._dist_args),
            },
        }

    def advance(self: TTimingModel, ttf: float | int) -> TTimingModel:
        # no-op
        return self

    def get_execution_time(self) -> float:
        return float(self._dist.rvs())

    def get_expected_execution_time(self) -> float:
        return self._dist.expect()

    def get_mean_execution_time(self) -> float:
        return self._dist.mean()

    def state_info(self) -> Dict[str, Any]:
        return {}

    def reset(self) -> None:
        # no-op
        pass

    def get_cdf_at_instant(self, instant: float) -> float:
        return float(self._dist.cdf(instant))


class FirstOrderAggregateETM(FirstOrderETM):
    def __init__(self, aggregate_fn: Callable[[npt.NDArray], float]):
        super().__init__()
        self._agg_exec_time = aggregate_fn(self._exec_times)

    def get_execution_time(self) -> float:
        return self._agg_exec_time

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "execution_time_seconds": self._agg_exec_time,
        }

    def get_expected_execution_time(self) -> float:
        return self._agg_exec_time

    def get_cdf_at_instant(self, instant: float) -> float:
        return float(instant < self._agg_exec_time)
