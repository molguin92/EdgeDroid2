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
import pandas as pd
from numpy import typing as npt
from scipy import stats

from .base import TTimingModel, ExecutionTimeModel


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

    @staticmethod
    def get_data() -> Any:
        import edgedroid.data as e_data

        return e_data.load_default_exec_time_data()

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


class LegacyModel(ExecutionTimeModel):
    @staticmethod
    def get_data() -> pd.DataFrame:
        import edgedroid.data as e_data

        return e_data.load_curve_fitting_data()

    def __init__(self, seed: int = 4):  # https://xkcd.com/221/
        super().__init__()
        rng = np.random.default_rng(seed)

        data = self.get_data()
        data = data[data["prev_ttf"] == data["prev_ttf"].min()].copy()

        self.times = (
            data.groupby(["prev_duration"], observed=True)["exec_time"]
            .apply(lambda a: rng.choice(a))
            .reset_index()
        )  # one execution time per duration

        self._current_duration = 0

    def advance(self: TTimingModel, ttf: float | int) -> TTimingModel:
        self._current_duration += 1
        return self

    def get_execution_time(self) -> float:
        return self.times.loc[
            self.times["prev_duration"].array.contains(self._current_duration),
        ].iat[0, 1]

    def get_expected_execution_time(self) -> float:
        return self.get_execution_time()

    def get_mean_execution_time(self) -> float:
        return self.get_execution_time()

    def get_cdf_at_instant(self, instant: float):
        raise Exception("Not implemented yet!")

    def state_info(self) -> Dict[str, Any]:
        raise Exception("Not implemented yet!")

    def reset(self) -> None:
        self._current_duration = 0

    def get_model_params(self) -> Dict[str, Any]:
        raise Exception("Not implemented yet!")
