from typing import Dict, Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from scipy.optimize import curve_fit

from .base import ExecutionTimeModel, TTimingModel, ModelException


class CurveFit:
    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        func: Callable[[npt.ArrayLike, ...], npt.ArrayLike],
        func_name: str,
    ):
        self._fname = func_name
        self._opt, _, info, *_ = curve_fit(func, x, y, full_output=True)
        self._func = func
        self._err = info["fvec"]
        self._mse = np.mean(np.square(self._err))

    def y(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return self._func(x, *self._opt)

    def __str__(self) -> str:
        return self._fname

    @property
    def name(self) -> str:
        return str(self)

    @property
    def error(self) -> npt.ArrayLike:
        return self._err

    @property
    def mse(self) -> float:
        return self._mse


class PowerFit(CurveFit):
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike):
        super().__init__(
            x,
            y,
            func=lambda x, a, b, c: a * np.power(x, b) + c,
            func_name="a * x^b + c",
        )


class SquareFit(CurveFit):
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike):
        super().__init__(
            x,
            y,
            func=lambda x, a, b, c: a * np.square(x) + b * x + c,
            func_name="a * x^2 + b * x + c",
        )


class CubeFit(CurveFit):
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike):
        super().__init__(
            x,
            y,
            func=lambda x, a, b, c, d: a * np.power(x, 3)
            + b * np.square(x)
            + c * x
            + d,
            func_name="a * x^3 + b * x^2 + c * x + d",
        )


class ExponentialFit(CurveFit):
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike):
        super().__init__(
            x,
            y,
            func=lambda x, a, b: a * np.exp(x) + b,
            func_name="a * e^x + b",
        )


class LinearFit(CurveFit):
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike):
        super().__init__(
            x,
            y,
            func=lambda x, a, b: a * x + b,
            func_name="a * x + b",
        )


def outliers_to_nan_then_mean(a: npt.NDArray | npt.ArrayLike, p: float = 5) -> float:
    upper_bound = np.percentile(a, 100 - p)
    lower_bound = np.percentile(a, p)
    a[a > upper_bound] = np.nan
    a[a < lower_bound] = np.nan
    return np.mean(a)


class MultiCurveFittingExecutionTimeModel(ExecutionTimeModel):
    _fit_functions = (PowerFit, SquareFit, CubeFit, ExponentialFit, LinearFit)

    @staticmethod
    def get_data() -> pd.DataFrame:
        import edgedroid.data as e_data

        return e_data.load_curve_fitting_data()

    def __init__(
        self,
        neuroticism: float,
        duration_reset_threshold: float = 0.5,
        agg_fn: Callable[[npt.ArrayLike], float] = outliers_to_nan_then_mean,
    ):
        logger.debug(f"Curve fitting aggregation function: {agg_fn.__name__}")
        self._duration_reset = duration_reset_threshold

        curve_data = self.get_data()

        # filter on our level of neuroticism, then group and aggregate
        curve_data = curve_data[curve_data["neuro"].array.contains(neuroticism)]

        curve_agg = (
            curve_data.groupby(["prev_duration", "prev_ttf"], observed=True)[
                "exec_time"
            ]
            .aggregate(agg_fn)
            .reset_index()
        )

        self._max_ttf = curve_agg["prev_ttf"].max()
        self._min_ttf = curve_agg["prev_ttf"].min()

        # fit a curve for each duration
        self._current_duration = 1
        self._current_duration_func = None
        self._prev_ttf = 0.0
        self._exec_time_funcs: Dict[pd.Interval, CurveFit] = dict()

        logger.info("Fitting execution time functions to data...")

        for duration, df in curve_agg.groupby("prev_duration", observed=True):
            mse = np.inf
            for fn_cls in self._fit_functions:
                fn = fn_cls(df["prev_ttf"], df["exec_time"])
                if fn.mse < mse:
                    mse = fn.mse
                    logger.info(
                        f"New best fit function for duration {duration}: {fn} "
                        f"(MSE: {fn.mse:0.2f}, prev. MSE {mse:0.2f})"
                    )
                    self._exec_time_funcs[duration] = fn

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
        if abs(ttf - self._prev_ttf) >= self._duration_reset:
            self._current_duration = 1
        else:
            self._current_duration += 1

        self._update_duration_func()

        self._prev_ttf = np.clip(ttf, a_min=self._min_ttf, a_max=self._max_ttf)
        return self

    def get_execution_time(self) -> float:
        return self._current_duration_func.y(self._prev_ttf)

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
