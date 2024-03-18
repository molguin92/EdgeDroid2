#!/usr/bin/env python3

import asyncio
import time
from pathlib import Path

import click


async def run_emulation_pair(
    sem: asyncio.Semaphore, experiment: str, output_dir: Path, truncate: int
):
    async with sem:
        proc = await asyncio.create_subprocess_shell(
            f"docker compose -p {str(output_dir).replace('/', '_')} up",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            env={
                "TRUNCATE": str(truncate),
                "EXPERIMENT": experiment,
                "OUTPUT": str(output_dir.resolve()),
            },
        )
        try:
            assert await proc.wait() == 0
        except AssertionError:
            print(await proc.stdout.read())
            print(await proc.stderr.read())
            raise


async def amain(
    parallel: int, reps: int, output_dir: Path, experiment: str, truncate: int
) -> None:
    async with asyncio.TaskGroup() as tg:
        sem = asyncio.Semaphore(value=parallel)
        for i in range(reps):
            task_output = output_dir / f"run{i:03d}"
            tg.create_task(run_emulation_pair(sem, experiment, task_output, truncate))


@click.command
@click.argument("experiment", type=str)
@click.option("-r", "--reps", type=int, default=5, show_default=True)
@click.option(
    "-o",
    "--output_dir",
    type=click.Path(exists=False, path_type=Path),
    show_default=True,
)
@click.option("-p", "--parallel", type=int, default=3, show_default=True)
@click.option("-t", "--truncate", type=int, default=10, show_default=True)
def main(experiment: str, reps: int, output_dir: Path, truncate: int, parallel: int):

    actual_output = output_dir / f"{experiment}_{int(time.time())}"
    actual_output.mkdir(exist_ok=True, parents=True)
    asyncio.run(amain(parallel, reps, actual_output, experiment, truncate))


if __name__ == "__main__":
    main()
