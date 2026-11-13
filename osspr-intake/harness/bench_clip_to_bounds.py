"""Time clip_to_bounds on a training matrix with shared per-feature bounds.

Models and tools clip every row before a private query. Equal per-feature
bounds take the scalar np.clip path after an explicit copy.
"""
import gc
import time

import numpy as np

from diffprivlib.validation import clip_to_bounds

_RNG = np.random.default_rng(0)
_ARRAY = _RNG.normal(size=(2_000_000, 16))
_BOUNDS = (np.full(16, -3.0), np.full(16, 3.0))


def workload():
    return clip_to_bounds(_ARRAY, _BOUNDS)


def main():
    for _ in range(4):
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
