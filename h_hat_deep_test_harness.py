import math
import os
import random
import statistics
from typing import Dict, List

from h_hat_predictor import HHatPredictor
from h_hat_test_harness import generate_random_params


_DEF_NUM_SAMPLES = 1000


def run_deep_dive(num_samples: int = _DEF_NUM_SAMPLES) -> None:
    """Run an extensive randomised Monte-Carlo simulation and print summary stats.

    Parameters
    ----------
    num_samples:
        Number of synthetic parameter sets to evaluate.  The higher the number,
        the better the approximation of the response distribution.
    """
    predictor = HHatPredictor()

    params_batch: List[Dict[str, float]] = generate_random_params(num_samples)
    predictions = [predictor.predict_from_params(p) for p in params_batch]

    print("=" * 72)
    print(f"Deep-dive statistics over {num_samples:,} simulated samples")
    print("=" * 72)
    print(f"min   : {min(predictions):.6f}")
    print(f"max   : {max(predictions):.6f}")
    print(f"mean  : {statistics.mean(predictions):.6f}")
    print(f"median: {statistics.median(predictions):.6f}")
    print(f"stdev : {statistics.pstdev(predictions):.6f}")


if __name__ == "__main__":
    run_deep_dive()