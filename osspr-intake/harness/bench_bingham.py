"""Time Bingham.randomise on an 8-dimensional covariance.

covariance_eig and PCA draw a private eigenvector with Bingham. The matrix
here is a fixed symmetric Gram matrix, and the seed is fixed, so each call
follows the same rejection path.
"""
import gc
import time

import numpy as np

from diffprivlib.mechanisms import Bingham

_RNG = np.random.default_rng(0)
_FACTOR = _RNG.normal(size=(8, 8))
_MATRIX = _FACTOR.T @ _FACTOR


def workload():
    mechanism = Bingham(epsilon=1.0, sensitivity=1.0, random_state=0)
    return mechanism.randomise(_MATRIX)


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
