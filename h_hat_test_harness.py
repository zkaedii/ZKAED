import math
import os
import random
from typing import Dict, List

import numpy as np

from h_hat_predictor import HHatPredictor


_RNG = random.Random()

# ---------------------------------------------------------------------------
# Synthetic parameter generation helpers
# ---------------------------------------------------------------------------

# NOTE: These ranges are *placeholder* values because the real feature space is
# unknown.  Update them to match your domain.
_FEAT_RANGES: Dict[str, tuple[float, float]] = {
    "mean": (-10.0, 10.0),
    "std": (0.0, 5.0),
    "min": (-20.0, 0.0),
    "max": (0.0, 20.0),
    "mean_first_derivative": (-5.0, 5.0),
    "std_first_derivative": (0.0, 50.0),
    "mean_second_derivative": (-20.0, 20.0),
    "std_second_derivative": (0.0, 500.0),
    "mean_jerk": (-1500.0, 1500.0),
    "std_jerk": (0.0, 500000.0),
}


def _sample_param_set() -> Dict[str, float]:
    return {
        name: _RNG.uniform(low, high)  # noqa: S311 – deterministic RNG wrapper
        for name, (low, high) in _FEAT_RANGES.items()
    }


def generate_random_params(batch_size: int = 1) -> List[Dict[str, float]]:
    """Generate *batch_size* random parameter dictionaries."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    return [_sample_param_set() for _ in range(batch_size)]


# ---------------------------------------------------------------------------
# Test execution helpers
# ---------------------------------------------------------------------------


def run_test_cases(num_cases: int = 10, verbose: bool = True) -> None:
    """Run *num_cases* randomised predictions using the default model.

    The intention is to provide a *smoke test* that the model file can be loaded
    and invoked end-to-end.  The function prints each prediction when *verbose*
    is *True*.
    """
    predictor = HHatPredictor()

    cases = generate_random_params(num_cases)
    preds = [predictor.predict_from_params(p) for p in cases]

    if verbose:
        for idx, (params, pred) in enumerate(zip(cases, preds), start=1):
            print(f"CASE {idx:02d}: prediction={pred!r} | params={params}")


if __name__ == "__main__":
    # Execute harness when run via `python -m h_hat_test_harness`
    run_test_cases()