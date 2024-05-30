from typing import Iterator

import scipy as sp

__all__ = ["gen_autocorrel"]


def gen_autocorrel(
    dist: sp.stats.rv_continuous, loc: float, rho: float, **kwargs
) -> Iterator[float]:

    loc = loc * (1 / rho)
    while True:
        value = dist.rvs(loc=(loc * rho), size=1, **kwargs).take(0)
        yield value
        loc = value
