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
from dataclasses import asdict, dataclass
from typing import List, Optional

from .sampling import *
from .timings import *


@dataclass(frozen=True)
class ModelFrame:
    seq: int  # absolute seq number
    step_seq: int  # seq number for current step
    step_index: int
    step_frame_time: float
    step_target_time: float
    frame_tag: str
    frame_data: npt.NDArray
    extra_data: Dict[str, Any]


@dataclass(frozen=True, eq=True)
class StepRecord:
    step_number: int
    last_frame_rtt: float
    execution_time: float
    step_duration: float
    time_to_feedback: float  # difference between step duration and execution time
    wait_time: float  # time between execution time and when final sample is taken
    frame_count: int

    def to_dict(self) -> Dict[str, int | float]:
        return asdict(self)


class EdgeDroidModel:
    """
    Implements the full end-to-end emulation of a human user in Cognitive
    Assistance.
    """

    def __init__(
        self,
        frame_trace: sampling.FrameSet,
        timing_model: timings.ExecutionTimeModel,
        frame_model: sampling.BaseSamplingPolicy,
    ):
        """
        Parameters
        ----------
        frame_trace
            A FrameSet object containing the video frame trace for the
            target task.
        timing_model
            An ExecutionTimeModel object to provide the timing information.
        frame_model
            A BaseFrameSamplingModel object to provide frame distribution information at
            runtime.
        """
        super(EdgeDroidModel, self).__init__()

        self._timings = timing_model
        self._frames = frame_trace
        self._frame_dists = frame_model
        self._frame_count = 0
        self._step_records: List[StepRecord] = []

    def reset(self) -> None:
        """
        Resets this model.
        """
        self._step_records.clear()
        self._timings.reset()
        self._frame_count = 0

    def model_step_metrics(self) -> pd.DataFrame:
        return pd.DataFrame(
            [a.to_dict() for a in self._step_records],
        ).set_index("step_number")

    def timing_model_params(self) -> Dict[str, Any]:
        return self._timings.get_model_params()

    @property
    def step_count(self) -> int:
        return self._frames.step_count

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def play_steps_realtime(
        self,
    ) -> Iterator[Generator[ModelFrame, FrameTimings, None]]:
        import time

        for step in self.play_steps_emulated_time():

            def _step_iter() -> Generator[ModelFrame, FrameTimings, None]:
                timings: Optional[FrameTimings] = None

                while True:
                    ti = time.monotonic()
                    try:
                        timings = yield step.send(timings)
                    except StopIteration:
                        break

                    dt = time.monotonic() - ti
                    assert dt >= timings.get_rtt()
                    assert dt >= timings.proctime_s

                    timings = FrameTimings(dt - timings.proctime_s, timings.proctime_s)

            yield _step_iter()

    def play_steps_emulated_time(
        self,
    ) -> Iterator[Generator[ModelFrame, FrameTimings, None]]:
        self.reset()

        class StepMetrics:
            model = self

            def __init__(self):
                self.frame_count = 0
                self.last_frame: Optional[FrameTimings] = None
                self.duration = 0.0

            def advance_frame(self, frame: FrameTimings):
                self.frame_count += 1
                self.model._frame_count += 1
                self.last_frame = frame
                self.duration += frame.nettime_s + frame.proctime_s

        step_metrics = StepMetrics()

        def _init_iter() -> Generator[ModelFrame, FrameTimings, None]:
            timings = yield ModelFrame(
                seq=self._frame_count,
                step_seq=1,
                step_index=-1,
                step_target_time=0,
                step_frame_time=0,
                frame_tag="initial",
                frame_data=self._frames.get_initial_frame(),
                extra_data={},
            )
            step_metrics.advance_frame(timings)

        yield _init_iter()

        step_record = StepRecord(
            step_number=0,
            last_frame_rtt=step_metrics.duration,
            execution_time=0.0,
            step_duration=step_metrics.duration,
            time_to_feedback=step_metrics.duration,
            wait_time=0.0,
            frame_count=step_metrics.frame_count,
        )
        self._step_records.append(step_record)

        for step_index in range(self.step_count):
            # get a step duration
            ttf = self._step_records[-1].time_to_feedback
            execution_time = self._timings.advance(ttf).get_execution_time()
            step_metrics = StepMetrics()

            def _frame_iter_for_step() -> Generator[ModelFrame, FrameTimings, None]:
                # play frames for step
                frame_iter = self._frame_dists.step_iterator(
                    target_time=execution_time,
                    ttf=ttf,
                )
                timings: Optional[FrameTimings] = None

                while True:
                    try:
                        sample = frame_iter.send(timings)
                    except StopIteration:
                        break

                    timings = yield ModelFrame(
                        seq=self._frame_count,
                        step_seq=sample.seq,
                        step_index=step_index,
                        step_frame_time=sample.instant,
                        step_target_time=execution_time,
                        frame_tag=sample.sample_tag,
                        frame_data=self._frames.get_frame(
                            step_index,
                            sample.sample_tag,
                        ),
                        extra_data=sample.extra,
                    )
                    step_metrics.advance_frame(timings)

            yield _frame_iter_for_step()

            last_frame = step_metrics.last_frame
            last_frame_rtt = last_frame.proctime_s + last_frame.nettime_s
            time_to_feedback = step_metrics.duration - execution_time
            wait_time = time_to_feedback - last_frame_rtt

            step_record = StepRecord(
                step_number=step_index + 1,
                last_frame_rtt=last_frame_rtt,
                execution_time=execution_time,
                step_duration=step_metrics.duration,
                time_to_feedback=time_to_feedback,
                wait_time=wait_time,
                frame_count=step_metrics.frame_count,
            )

            self._step_records.append(step_record)
