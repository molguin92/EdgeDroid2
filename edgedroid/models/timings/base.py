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
from typing import TypeVar, Iterator, Tuple, Dict, Any

import pandas as pd
from pandas import arrays


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


