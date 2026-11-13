"""Time GeometricTruncated.randomise on a fine histogram.

histogram and histogramdd add one geometric draw per bin, reusing a single
mechanism. The timed call is that loop: several thousand integer bin counts.
"""
import gc
import sys
import time

import numpy as np

from diffprivlib.mechanisms import GeometricTruncated

_RNG = np.random.default_rng(0)
_SAMPLE = _RNG.random(200_000)
_COUNTS, _ = np.histogram(_SAMPLE, bins=12_000, range=(0.0, 1.0))


def workload():
    mechanism = GeometricTruncated(
        epsilon=1.0, sensitivity=1, lower=0, upper=sys.maxsize, random_state=0
    )
    total = 0
    for count in _COUNTS:
        total += mechanism.randomise(int(count))
    return total


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
