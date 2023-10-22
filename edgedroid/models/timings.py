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
import copy
import enum
from collections import deque
from typing import Any, Dict, Iterator, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import arrays
from scipy import stats


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


class ModelException(Exception):
    """
    Exception raised during model execution.
    """

    pass


class Transition(str, enum.Enum):
    H2L = "Higher2Lower"
    L2H = "Lower2Higher"
    NONE = "NoTransition"


def preprocess_data(
    exec_time_data: pd.DataFrame,
    neuro_bins: arrays.IntervalArray | pd.IntervalIndex,
    impair_bins: arrays.IntervalArray | pd.IntervalIndex,
    duration_bins: arrays.IntervalArray | pd.IntervalIndex,
    # transition_fade_distance: Optional[int] = None,
) -> pd.DataFrame:
    """
    Processes a DataFrame with raw execution time data into a DataFrame
    usable by the model.

    The argument DataFrame must be in order (of steps) and have the following
    columns:

    - run_id (categorical or int)
    - neuroticism (float)
    - exec_time (float)
    - ttf (float)

    Parameters
    ----------
    exec_time_data
        Raw experimental data
    neuro_bins
        Bins to use for neuroticism values.
    impair_bins
        Bins to use for time-to-feedback (impairment).
    duration_bins
        Bins to use for sequences of same impairment.

    Returns
    -------
        A DataFrame.
    """

    proc_data = exec_time_data.copy()

    for col in ("run_id", "neuroticism", "exec_time", "ttf"):
        if col not in proc_data.columns:
            raise ModelException(f"Base data missing required column: {col}")

    proc_data["neuroticism_raw"] = proc_data["neuroticism"]
    proc_data["neuroticism"] = pd.cut(
        proc_data["neuroticism"], pd.IntervalIndex(neuro_bins)
    )

    processed_dfs = deque()
    for run_id, df in proc_data.groupby("run_id"):
        df = df.copy()
        df["ttf"] = df["ttf"].shift().fillna(0)

        df["impairment"] = pd.cut(df["ttf"], pd.IntervalIndex(impair_bins))
        df = df.rename(columns={"exec_time": "next_exec_time"})

        # df["next_exec_time"] = df["exec_time"].shift(-1)
        df["prev_impairment"] = df["impairment"].shift()
        # df["transition"] = Transition.NONE.value

        # for each segment with the same impairment, count the number of steps
        # (starting from 1)
        diff_imp_groups = df.groupby(
            (df["impairment"].ne(df["prev_impairment"])).cumsum()
        )
        df["duration"] = diff_imp_groups.cumcount() + 1

        df["transition"] = None
        df.loc[
            df["prev_impairment"] < df["impairment"], "transition"
        ] = Transition.L2H.value
        df.loc[
            df["prev_impairment"] > df["impairment"], "transition"
        ] = Transition.H2L.value

        df["transition"] = (
            df["transition"].fillna(method="ffill").fillna(Transition.NONE.value)
        )

        processed_dfs.append(df)

    proc_data = pd.concat(processed_dfs, ignore_index=False)

    # coerce some types for proper functionality
    proc_data["transition"] = proc_data["transition"].astype("category")
    proc_data["neuroticism"] = proc_data["neuroticism"].astype(pd.IntervalDtype(float))
    proc_data["impairment"] = proc_data["impairment"].astype(pd.IntervalDtype(float))
    proc_data["duration_raw"] = proc_data["duration"]
    proc_data["duration"] = pd.cut(
        proc_data["duration"], pd.IntervalIndex(duration_bins)
    ).astype(pd.IntervalDtype(float))
    proc_data = proc_data.drop(columns="prev_impairment")

    return proc_data


# workaround for typing methods of classes as returning the same type as the
# enclosing class, while also working for extending classes
TTimingModel = TypeVar("TTimingModel", bound="ExecutionTimeModel")


class ExecutionTimeModel(Iterator[float], metaclass=abc.ABCMeta):
    """
    Defines the general interface for execution time models.
    """

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

        return e_data.load_default_exec_time_data()

    def __iter__(self):
        return self

    def __next__(self) -> float:
        return self.get_execution_time()

    @abc.abstractmethod
    def advance(self: TTimingModel, ttf: float | int) -> TTimingModel:
        """
        Update the internal TTF of this model and advance the internal state.

        Parameters
        ----------
        ttf
            Time-to-feedback of the previous step, expressed in seconds.
        """
        return self

    @abc.abstractmethod
    def get_execution_time(self) -> float:
        """
        Obtain an execution time from this model.

        Returns
        -------
        float
            An execution time value in seconds.
        """
        pass

    @abc.abstractmethod
    def get_expected_execution_time(self) -> float:
        """
        Returns the *expected* execution time for the current state of the model.

        Returns
        -------
        float
            An execution time value in seconds.
        """
        pass

    @abc.abstractmethod
    def get_mean_execution_time(self) -> float:
        pass

    @abc.abstractmethod
    def get_cdf_at_instant(self, instant: float):
        """
        Returns the value of the CDF for the execution time distribution of the
        current state of the model at the given instant.

        Parameters
        ----------
        instant: float

        Returns
        -------
        float
        """
        pass

    @abc.abstractmethod
    def state_info(self) -> Dict[str, Any]:
        """
        Returns
        -------
        dict
            A dictionary containing debugging information about the internal
            stated of this model.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Resets the internal state to the starting state.
        """
        pass

    def copy(self: TTimingModel) -> TTimingModel:
        """
        Returns a (deep) copy of this model.
        """
        return copy.deepcopy(self)

    def fresh_copy(self: TTimingModel) -> TTimingModel:
        """
        Returns a (deep) copy of this model, reset to its initial state.
        """
        model = self.copy()
        model.reset()
        return model

    @abc.abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        pass


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


def _convolve_kernel(arr: pd.Series, kernel: npt.NDArray):
    index = arr.index
    arr = arr.to_numpy()
    arr = np.concatenate([np.zeros(kernel.size) + arr[0], arr])
    lkernel = np.concatenate([np.zeros(kernel.size - 1), kernel / kernel.sum()])
    result = np.convolve(arr, lkernel, "same")
    return pd.Series(result[kernel.size :], index=index)


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


class CleanupMode(enum.Enum):
    NONE = enum.auto()
    WINSORIZE = enum.auto()
    TRUNCATE = enum.auto()


class EmpiricalETM(ExecutionTimeModel):
    @staticmethod
    def make_kernel(window: int, exp_factor: float = 0.7):
        kernel = np.zeros(window)
        for i in range(window):
            kernel[i] = np.exp(-exp_factor * i)

        return kernel / kernel.sum()

    def __init__(
        self,
        neuroticism: float | None,
        window: int = 12,
        ttf_levels: int = 4,
        cleanup: CleanupMode = CleanupMode.WINSORIZE,
    ):
        data, neuro_bins, *_ = self.get_data()

        data["next_exec_time"] = data["exec_time"].shift(-1)
        data = data.dropna()

        # roll the ttfs
        self._kernel = self.make_kernel(window)
        data["rolling_ttf"] = (
            data.groupby("run_id")["ttf"]
            .apply(lambda arr: _convolve_kernel(arr, self._kernel))
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

        self._window = np.zeros(window, dtype=float)
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
        weighted_ttf = np.multiply(self._window, self._kernel).sum()
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
            "weights": self._kernel,
            "weighted_ttf": np.multiply(self._window, self._kernel).sum(),
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
        }

    def get_cdf_at_instant(self, instant: float) -> float:
        exec_times = self._views[self._get_binned_ttf()]
        return exec_times[exec_times < instant].size / exec_times.size


class FittedETM(EmpiricalETM):
    def __init__(
        self,
        neuroticism: float | None,
        dist: stats.rv_continuous = stats.exponnorm,
        window: int = 12,
        ttf_levels: int = 4,
        cleanup: CleanupMode = CleanupMode.WINSORIZE,
    ):
        super(FittedETM, self).__init__(
            neuroticism=neuroticism,
            window=window,
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


class LegacyETM(EmpiricalETM):
    """
    EdgeDroid 1.0 execution time model.
    """

    def __init__(
        self,
        seed: int = 4,
        window: int = 12,
        ttf_levels: int = 7,
    ):
        super(LegacyETM, self).__init__(
            neuroticism=0.0,
            window=window,
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
