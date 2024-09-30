import itertools as it
import warnings
from typing import Tuple

import multiprocess as mp
import scipy.stats as st
from generate_corr_sequence import gen_corr_sequence
from nptyping.ndarray import NDArray
from scipy.integrate import IntegrationWarning
from tqdm import tqdm

import edgedroid.data as e_data
from edgedroid.models import *


def lognorm_with_params(
    loc: float | int, mean: float | int, var: float | int
) -> Tuple[stats.rv_continuous, float, float]:
    sigma2 = np.log((var / np.square(mean - loc)) + 1)
    mu = np.log(mean - loc) - (sigma2 / 2)
    scale = np.exp(mu)
    s = np.sqrt(sigma2)
    return st.lognorm(loc=loc, scale=scale, s=s), scale, s


def gamma_with_params(
    loc: float | int, mean: float | int, var: float | int
) -> Tuple[stats.rv_continuous, float, float]:
    scale = var / (mean - loc)
    k = np.square(mean - loc) / var
    return st.gamma(a=k, loc=loc, scale=scale), scale, k


class ThirdDegPolyExecutionTimeModel(MultiCurveFittingExecutionTimeModel):
    _fit_functions = (CubeFit,)


def emulate_run_with_frame_delays(model: EdgeDroidModel, rtt_s_iter: Iterator[float]):
    for model_step in model.play_steps_emulated_time():
        # in this setup, we have rtts per frame
        frame_timings: Optional[FrameTimings] = None
        prev_frame: Optional[str] = None
        while True:
            try:
                model_frame = model_step.send(frame_timings)
            except StopIteration:
                if not prev_frame or prev_frame.lower() not in ("success", "initial"):
                    raise Exception(prev_frame)
                break

            # prev_result = task.submit_frame(model_frame.frame_data)
            prev_frame = model_frame.frame_tag
            frame_rtt = next(rtt_s_iter)
            frame_timings = FrameTimings(frame_rtt * 0.9, frame_rtt * 0.1)


def emulate_run_with_step_delays(model: EdgeDroidModel, rtt_s_iter: Iterator[float]):
    for model_step in model.play_steps_emulated_time():
        # in this setup, we have fixed rtts per step
        step_rtt = next(rtt_s_iter)

        frame_timings: Optional[FrameTimings] = None
        prev_frame: Optional[str] = None
        while True:
            try:
                model_frame = model_step.send(frame_timings)
            except StopIteration:
                if not prev_frame or prev_frame.lower() not in ("success", "initial"):
                    raise Exception(prev_frame)
                break

            # prev_result = task.submit_frame(model_frame.frame_data)
            prev_frame = model_frame.frame_tag
            frame_timings = FrameTimings(step_rtt * 0.9, step_rtt * 0.1)


def run_model(params):
    # print(params)
    (
        (start_window, rep, delays, loc, mean, var, scale, shape, rho),
        (model_name, model_constructor),
        (experiment_name, emulation_fn),
    ) = params
    timing_model, sampling_policy = model_constructor()

    model = EdgeDroidModel(
        frame_trace=e_data.load_default_trace("square00", truncate=50),
        frame_model=sampling_policy,
        timing_model=timing_model,
    )

    delays = delays[np.random.randint(start_window) :]
    emulation_fn(model, iter(delays))

    df = model.model_step_metrics()
    df["experiment"] = experiment_name
    df["model"] = model_name
    df["rep"] = rep
    df["rho"] = rho
    df["gamma_loc"] = loc
    df["gamma_scale"] = scale
    df["gamma_shape"] = shape
    df["gamma_mean"] = mean
    df["gamma_var"] = var

    return df


def precalculate_delays(
    params: tuple[int, float, float, float, float, Optional[NDArray], int]
):
    (rep, loc, mean, var, rho, acf, count) = params
    dist, scale, shape = gamma_with_params(loc=loc, mean=mean, var=var)

    if acf is not None:
        delays = gen_corr_sequence(
            dist_obj=dist,
            L=count,
            target_acf=acf,
            debug=False,
        )
    else:
        delays = dist.rvs(size=count)
    return rep, delays, loc, mean, var, scale, shape, rho


def estimate_std(mean: float, jitter: float) -> float:
    samples = np.empty(1_000_000)
    for i in range(samples.size):
        samples[i] = mean + ((np.random.random() * 2 * jitter) - jitter)

    return samples.std()


if __name__ == "__main__":
    # warnings.filterwarnings("ignore", category=IntegrationWarning)

    reps_per_model = 30

    acf_50 = 1 / (2 ** np.arange(5))  # 0.5 corr
    acf_25 = 1 / (4 ** np.arange(5))
    acf_12 = 1 / (8 ** np.arange(5))

    num_frames_to_generate = 6 * 3600 * 30  # generate 6 hours of delays
    start_window = 2 * 3600 * 30  # start window is within the first two hours

    timing_frame_models = {
        "3rd-poly-high": lambda: (
            ThirdDegPolyExecutionTimeModel(neuroticism=1.0),
            ZeroWaitSamplingPolicy.from_default_data(),
        ),
        "3rd-poly-low": lambda: (
            ThirdDegPolyExecutionTimeModel(neuroticism=0.0),
            ZeroWaitSamplingPolicy.from_default_data(),
        ),
        "legacy": lambda: (LegacyModel(), LegacySamplingPolicy.from_default_data()),
        "first-order": lambda: (
            FirstOrderETM(),
            ZeroWaitSamplingPolicy.from_default_data(),
        ),
        "first-order-median": lambda: (
            FirstOrderAggregateETM(np.median),
            ZeroWaitSamplingPolicy.from_default_data(),
        ),
    }

    min_bound = 0.042  # 24FPS

    rhos = (
        (0.0, None),
        # (0.125, acf_12),
        (0.250, acf_25),
        (0.500, acf_50),
    )

    mean_jitter = (
        (0.10, 0.02),
        (0.20, 0.04),
        (0.40, 0.08),
        (0.80, 0.16),
        (1.60, 0.32),
        (3.20, 0.64),
    )

    mean_vars = [
        (mean, estimate_std(mean, jitter) ** 2) for mean, jitter in mean_jitter
    ]

    delay_params = [
        (rep, min_bound, mean, var, rho, acf, num_frames_to_generate)
        for rep, (mean, var), (rho, acf) in it.product(
            range(reps_per_model), mean_vars, rhos
        )
    ]

    with mp.Pool() as pool:
        # precalculate delays
        delays = deque()
        with tqdm(total=len(delay_params), desc="Preparing delays") as bar:
            for d in pool.imap_unordered(precalculate_delays, delay_params):
                delays.append((start_window, *d))
                bar.update(1)

        params_iter = list(
            it.product(
                delays,
                timing_frame_models.items(),
                (
                    ("frame", emulate_run_with_frame_delays),
                    ("step", emulate_run_with_step_delays),
                ),
            )
        )
        dfs = deque()
        with tqdm(total=len(params_iter), desc="Running models") as bar:
            for df in pool.imap_unordered(run_model, params_iter):
                dfs.append(df)
                bar.update(1)

    data = pd.concat(dfs)
    data.to_csv("./per_frame_step_delay_new_gamma_params.csv")
