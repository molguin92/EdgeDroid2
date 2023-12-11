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

import abc
import enum
from typing import Dict, Any, Callable, Tuple

import numpy as np
import pandas as pd
from numpy import typing as npt
from scipy import stats

from .base import ExecutionTimeModel, TTimingModel, ModelException


class CleanupMode(enum.Enum):
    NONE = enum.auto()
    WINSORIZE = enum.auto()
    TRUNCATE = enum.auto()


_default_window_size = 8
_default_ttf_bins = 6
_default_cleanup = CleanupMode.TRUNCATE


def _winsorize(
        arr: npt.NDArray, low_percentile: int = 5, high_percentile: int = 95
) -> npt.NDArray:
    low_bound = np.percentile(arr, low_percentile)
    high_bound = np.percentile(arr, high_percentile)

    arr[arr < low_bound] = low_bound
    arr[arr > high_bound] = high_bound

    return arr


def _truncate(
        arr: npt.NDArray, low_percentile: int = 5, high_percentile: int = 95
) -> npt.NDArray:
    low_bound = np.percentile(arr, low_percentile)
    high_bound = np.percentile(arr, high_percentile)

    return np.copy(arr[np.logical_and(arr >= low_bound, arr <= high_bound)])


def _serialize_interval(interval: pd.Interval) -> Dict[str, float | bool]:
    left_open = interval.open_left
    right_open = interval.open_right

    if left_open and right_open:
        closed = "neither"
    elif left_open:
        closed = "right"
    elif right_open:
        closed = "left"
    else:
        closed = "both"

    return {
        "left": float(interval.left),
        "right": float(interval.right),
        "closed": closed,
    }


class TTFWindowKernel(abc.ABC):
    @property
    @abc.abstractmethod
    def window_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def weights(self) -> npt.NDArray:
        pass

    def convolve(self, arr: pd.Series) -> pd.Series:
        kernel = self.weights
        index = arr.index
        arr = arr.to_numpy()
        arr = np.concatenate([np.zeros(kernel.size) + arr[0], arr])
        lkernel = np.concatenate([np.zeros(kernel.size - 1), kernel])
        result = np.convolve(arr, lkernel, "same")
        return pd.Series(result[kernel.size:], index=index)

    def weighted_average(self, window: npt.NDArray) -> float:
        return np.multiply(window, self.weights).sum()


class ExponentialTTFWindowKernel(TTFWindowKernel):
    def __init__(self, window_size: int, exp_factor: float = 0.7):
        kernel = np.zeros(window_size)
        for i in range(window_size):
            kernel[i] = np.exp(-exp_factor * i)

        self._kernel = kernel / kernel.sum()

    @property
    def window_size(self) -> int:
        return self._kernel.size

    @property
    def weights(self) -> npt.NDArray:
        return self._kernel


class LinearTTFWindowKernel(TTFWindowKernel):
    def __init__(self, window_size: int, max_relative_weight: float | int, min_relative_weight: float | int = 1.0):
        # formula for a line intersecting two points (x1 y1) (x2 y2):
        # (y1 - y2) / (x1 - x2) = slope
        # y1 - (x1 * slope) = c
        # here we can say x1 = 0, x2 = (window_size - 1)
        max_relative_weight = float(max_relative_weight)
        min_relative_weight = float(min_relative_weight)

        slope = (max_relative_weight - min_relative_weight) / (1 - window_size)
        c = max_relative_weight

        kernel = np.zeros(window_size)
        for x in range(window_size):
            kernel[x] = (x * slope) + c

        self._kernel = kernel / kernel.sum()

    @property
    def window_size(self) -> int:
        return self._kernel.size

    @property
    def weights(self) -> npt.NDArray:
        return self._kernel


class AverageTTFWindowKernel(TTFWindowKernel):
    def __init__(self, window_size: int):
        kernel = np.ones(window_size)
        self._kernel = kernel / kernel.sum()

    @property
    def weights(self) -> npt.NDArray:
        return self._kernel

    @property
    def window_size(self) -> int:
        return self._kernel.size


class SimpleTTFWindowKernel(TTFWindowKernel):
    def __init__(self, relative_weights: npt.NDArray):
        kernel = relative_weights.copy()
        self._kernel = kernel / kernel.sum()

    @property
    def weights(self) -> npt.NDArray:
        return self._kernel

    @property
    def window_size(self) -> int:
        return self._kernel.size


class EmpiricalETM(ExecutionTimeModel):
    def __init__(
            self,
            kernel: TTFWindowKernel,
            neuroticism: float | None,
            ttf_levels: int = 6,
            cleanup: CleanupMode = CleanupMode.TRUNCATE,
    ):
        data, neuro_bins, *_ = self.get_data()

        data["next_exec_time"] = data["exec_time"].shift(-1)
        data = data.dropna()

        # roll the ttfs
        self._kernel = kernel
        data["rolling_ttf"] = (
            data.groupby("run_id")["ttf"]
            .apply(kernel.convolve)
            .droplevel(axis=0, level=0)
        )
        _, ttf_bins = pd.qcut(data["rolling_ttf"], ttf_levels, retbins=True)
        ttf_bins[0], ttf_bins[-1] = -np.inf, np.inf
        self._ttf_bins = pd.IntervalIndex.from_breaks(ttf_bins, closed="right")
        data["binned_rolling_ttf"] = pd.cut(data["rolling_ttf"], bins=self._ttf_bins)

        if neuroticism is not None:
            # bin neuroticism
            data["binned_neuro"] = pd.cut(
                data["neuroticism"], bins=pd.IntervalIndex(neuro_bins)
            ).astype(pd.IntervalDtype(float))
            data = data[data["binned_neuro"].array.contains(neuroticism)].copy()

        # prepare views
        self._views: Dict[pd.Interval, npt.NDArray] = {}
        for binned_rolling_ttf, df in data.groupby("binned_rolling_ttf", observed=True):
            exec_times = df["next_exec_time"].to_numpy()

            if cleanup == CleanupMode.WINSORIZE:
                exec_times = _winsorize(exec_times)
            elif cleanup == CleanupMode.TRUNCATE:
                exec_times = _truncate(exec_times)

            self._views[binned_rolling_ttf] = exec_times

        self._window = np.zeros(kernel.window_size, dtype=float)
        self._steps = 0
        self._neuroticism = neuroticism
        self._rng = np.random.default_rng()

    def advance(self: TTimingModel, ttf: float | int) -> TTimingModel:
        if self._steps == 0:
            self._window[:] = ttf
            self._steps += 1
        else:
            self._window = np.roll(self._window, shift=1)
            self._window[0] = ttf
        return self

    def _get_binned_ttf(self) -> pd.Interval:
        weighted_ttf = self._kernel.weighted_average(self._window)
        return self._ttf_bins[self._ttf_bins.contains(weighted_ttf)][0]

    def get_execution_time(self) -> float:
        return self._rng.choice(self._views[self._get_binned_ttf()])

    def get_expected_execution_time(self) -> float:
        return self._views[self._get_binned_ttf()].mean()

    def get_mean_execution_time(self) -> float:
        return self.get_expected_execution_time()

    def state_info(self) -> Dict[str, Any]:
        return {
            "ttf_window": self._window,
            "weights": self._kernel.weights,
            "weighted_ttf": self._kernel.weighted_average(self._window),
            "neuroticism": self._neuroticism,
            "steps": self._steps,
        }

    def reset(self) -> None:
        self._window = np.zeros(self._window.size, dtype=float)
        self._rng = np.random.default_rng()
        self._steps = 0

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "neuroticism": self._neuroticism,
            "window": self._window.size,
            "ttf_levels": len(self._ttf_bins),
            "kernel": self._kernel.__class__.__name__,
        }

    def get_cdf_at_instant(self, instant: float) -> float:
        exec_times = self._views[self._get_binned_ttf()]
        return exec_times[exec_times < instant].size / exec_times.size


class EmpiricalAggregateETM(EmpiricalETM):
    def __init__(
            self,
            aggregate_fn: Callable[[npt.NDArray], float],
            kernel: TTFWindowKernel,
            neuroticism: float | None,
            ttf_levels: int = 6,
            cleanup: CleanupMode = CleanupMode.TRUNCATE,
    ):
        super().__init__(
            kernel=kernel,
            neuroticism=neuroticism,
            ttf_levels=ttf_levels,
            cleanup=cleanup,
        )
        self._agg_fn = aggregate_fn

    def get_execution_time(self) -> float:
        return self._agg_fn(self._views[self._get_binned_ttf()])

    def get_expected_execution_time(self) -> float:
        return self.get_execution_time()

    def state_info(self) -> Dict[str, Any]:
        return {
            "ttf_window": self._window,
            "weights": self._kernel.weights,
            "weighted_ttf": self._kernel.weighted_average(self._window),
            "neuroticism": self._neuroticism,
            "steps": self._steps,
            "agg_fn": self._agg_fn.__name__,
        }

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "neuroticism": self._neuroticism,
            "window": self._window.size,
            "ttf_levels": len(self._ttf_bins),
            "kernel": self._kernel.__class__.__name__,
            "agg_fn": self._agg_fn.__name__,
        }

    def get_cdf_at_instant(self, instant: float) -> float:
        exec_time = self._agg_fn(self._views[self._get_binned_ttf()])
        return float(instant < exec_time)


class FittedETM(EmpiricalETM):
    def __init__(
            self,
            kernel: TTFWindowKernel,
            neuroticism: float | None,
            dist: stats.rv_continuous = stats.exponnorm,
            ttf_levels: int = 6,
            cleanup: CleanupMode = CleanupMode.TRUNCATE,
    ):
        super(FittedETM, self).__init__(
            kernel=kernel,
            neuroticism=neuroticism,
            ttf_levels=ttf_levels,
            cleanup=cleanup,
        )

        self._dists: Dict[pd.Interval, stats.rv_continuous] = {}
        for ttf_bin, exec_times in self._views.items():
            *params, loc, scale = dist.fit(exec_times)
            self._dists[ttf_bin] = dist.freeze(
                *params,
                loc=loc,
                scale=scale,
            )

        self._distribution = dist

    def get_execution_time(self) -> float:
        return max(self._dists[self._get_binned_ttf()].rvs(), 0.0)

    def get_expected_execution_time(self) -> float:
        return self._dists[self._get_binned_ttf()].expect()

    def get_mean_execution_time(self) -> float:
        return self._dists[self._get_binned_ttf()].mean()

    def get_model_params(self) -> Dict[str, Any]:
        params = super(FittedETM, self).get_model_params()
        params["distribution"] = self._distribution.name
        return params

    def get_cdf_at_instant(self, instant: float) -> float:
        return float(self._dists[self._get_binned_ttf()].cdf(instant))


class CurveFittingExecutionTimeModel(ExecutionTimeModel):

    @staticmethod
    def _exec_time_func(x, a, b, c) -> float:
        return a * np.power(x, b) + c

    @staticmethod
    def get_data() -> (
            Tuple[
                pd.DataFrame,
                pd.arrays.IntervalArray,
                pd.arrays.IntervalArray,
                pd.arrays.IntervalArray,
            ]
    ):
        import edgedroid.data as e_data
        return e_data.load_curve_fitting_data(), None, None, None

    def __init__(self, neuroticism: float):
        import scipy.optimize as opt

        curve_data, *_ = self.get_data()

        # filter on our level of neuroticism
        curve_data = curve_data[curve_data["neuro"].array.contains(neuroticism)]
        self._max_ttf = curve_data["ttf"].max()

        # fit a curve for each duration
        self._current_duration = 1
        self._current_duration_func = None
        self._prev_ttf = 0.0
        self._exec_time_funcs: Dict[pd.Interval, Callable[[float], float]] = dict()
        for duration, df in curve_data.groupby("duration", observed=True):
            coefs, *_ = opt.curve_fit(self._exec_time_func, xdata=df["ttf"], ydata=df["exec_time"])
            self._exec_time_funcs[duration] = (
                lambda ttf: self._exec_time_func(ttf, *coefs)
            )
            if self._current_duration in duration:
                self._current_duration_func = self._exec_time_funcs[duration]

        if self._current_duration_func is None:
            raise ModelException(f"No data for duration {self._current_duration}!")

    def _update_duration_func(self):
        for duration, func in self._exec_time_funcs.items():
            if self._current_duration in duration:
                self._current_duration_func = self._exec_time_funcs[duration]
                return
        raise ModelException(f"No data for duration {self._current_duration}!")

    def advance(self: TTimingModel, ttf: float | int) -> TTimingModel:
        if abs(ttf - self._prev_ttf) >= 0.5:  # TODO parameterize?
            self._current_duration = 1
        else:
            self._current_duration += 1

        self._update_duration_func()
        self._prev_ttf = min(ttf, self._max_ttf)
        return self

    def get_execution_time(self) -> float:
        return self._current_duration_func(self._prev_ttf)

    def get_expected_execution_time(self) -> float:
        return self.get_execution_time()

    def get_mean_execution_time(self) -> float:
        return self.get_execution_time()

    def get_cdf_at_instant(self, instant: float):
        raise Exception("Not implemented yet!")

    def state_info(self) -> Dict[str, Any]:
        raise Exception("Not implemented yet!")

    def reset(self) -> None:
        self._prev_ttf = 0.0
        self._current_duration = 1
        self._update_duration_func()

    def get_model_params(self) -> Dict[str, Any]:
        raise Exception("Not implemented yet!")
