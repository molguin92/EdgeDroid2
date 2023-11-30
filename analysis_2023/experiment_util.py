import itertools
import multiprocess as mp
import uuid
from collections import deque
from dataclasses import dataclass, asdict
from typing import Iterable, Deque, Callable, Collection, TypeAlias, Tuple, Any, Optional

import pandas as pd

from edgedroid.models.timings import ExecutionTimeModel

ModelBuilder: TypeAlias = Callable[[], Tuple[str, ExecutionTimeModel]]
TTFSeqBuilder: TypeAlias = Callable[[], Iterable[float]]


@dataclass(frozen=True, eq=True)
class StepRow:
    run_id: str
    model_name: str
    prev_ttf: float
    step_index: int
    exec_time: float
    step_time: float
    task_duration: float


def simulated_task(
        model_builder: ModelBuilder,
        ttfs_sequence_builder: TTFSeqBuilder,
) -> pd.DataFrame:
    """
    Executes a simulated run with a sequence of TTFs.
    This function assumes an implicit 0-valued TTF before the first step.
    The final TTF produced by the TTF iterator is not used to generate a new step, but influences the step duration of the final step.
    """
    run_id = uuid.uuid4().hex
    model_name, model = model_builder()
    ttfs = ttfs_sequence_builder()

    steps: Deque[dict] = deque()

    total_duration = 0

    # first step has a previous TTF of 0
    prev_ttf = 0.0
    for step_idx, ttf in enumerate(ttfs):
        exec_time = model.get_execution_time()
        step_time = exec_time + (ttf / 2.0)
        total_duration += step_time

        steps.append(
            asdict(StepRow(
                run_id=run_id,
                model_name=model_name,
                prev_ttf=prev_ttf,
                step_index=step_idx,
                exec_time=exec_time,
                step_time=step_time,
                task_duration=total_duration,
            ))
        )

        model.advance(ttf)
        prev_ttf = ttf

    return pd.DataFrame(steps)


def _map_helper(args: Tuple[Any]):
    return simulated_task(*args)


def run_experiment(
        model_builders: Collection[ModelBuilder],
        ttf_seq_builder: TTFSeqBuilder,
        reps_per_setup: int,
        processes: Optional[int] = None,
) -> pd.DataFrame:
    tasks = itertools.chain.from_iterable(itertools.repeat(model_builders, reps_per_setup))
    with mp.Pool(processes=processes) as pool:
        return pd.concat(
            pool.imap_unordered(
                _map_helper,
                zip(
                    tasks,
                    itertools.repeat(ttf_seq_builder)
                )
            )
        )
