"""Time Exponential._check_utility_candidates_measure on a large quantile.

quantile builds one utility and one measure per order statistic, then
Exponential validates both lists at construction and again in randomise.
"""
import gc
import time

import numpy as np

from diffprivlib.mechanisms.exponential import Exponential

_K = 250_000
_QUANT = 0.5
_RNG = np.random.default_rng(0)
_VALUES = np.clip(_RNG.normal(size=_K), -5.0, 5.0)
_VALUES = np.append(_VALUES, [-5.0, 5.0])
_VALUES.sort()
_UTILITY = list(-np.abs(np.arange(0, _K + 1) - _QUANT * _K))
_MEASURE = list(np.diff(_VALUES))


def workload():
    # __init__ and randomise each call this check on the same lists.
    Exponential._check_utility_candidates_measure(_UTILITY, None, _MEASURE)
    return Exponential._check_utility_candidates_measure(_UTILITY, None, _MEASURE)


def main():
    workload()
    workload()
    samples = []
    for _ in range(7):
        gc.collect()
        started = time.perf_counter()
        workload()
        samples.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(samples)
    median = ordered[len(ordered) // 2]
    noise = (ordered[-1] - ordered[0]) / median * 100.0
    rendered = ",".join(f"{sample:.4f}" for sample in samples)
    print(f"median_ms={median:.4f} noise_pct={noise:.4f} samples_ms={rendered}")


if __name__ == "__main__":
    main()
