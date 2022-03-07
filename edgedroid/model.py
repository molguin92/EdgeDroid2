import time
from typing import Iterator

import nptyping as npt

from .execution_times import ExecutionTimeModel
from .frames import FrameModel, FrameSet


class EdgeDroidModel:
    def __init__(self,
                 frame_trace: FrameSet,
                 timing_model: ExecutionTimeModel,
                 frame_model: FrameModel):
        super(EdgeDroidModel, self).__init__()

        self._timings = timing_model
        self._frames = frame_trace
        self._frame_dists = frame_model

    @property
    def step_count(self) -> int:
        return self._frames.step_count

    def play(self) -> Iterator[npt.NDArray]:
        self._timings.reset()

        prev_step_end = time.monotonic()

        # yield the initial frame of the task
        prev_success = self._frames.get_initial_frame()
        yield prev_success

        # task is now running
        for step_index in range(self._frames.step_count):
            # get a step duration
            delay = time.monotonic() - prev_step_end
            self._timings.set_delay(delay)
            step_duration = self._timings.get_execution_time()

            # replay frames for step
            for frame_tag, instant in self._frame_dists.step_iterator(
                    target_time=step_duration):
                # FIXME: hardcoded string tags

                if frame_tag == 'repeat':
                    yield prev_success
                elif frame_tag == 'success':
                    prev_success = self._frames.get_frame(step_index, frame_tag)

                    prev_step_end = time.monotonic()
                    yield prev_success
                else:
                    yield self._frames.get_frame(step_index, frame_tag)
