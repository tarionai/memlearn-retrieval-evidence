"""Public slice of the private ``tournament_base`` module.

PROVENANCE DISCLOSURE — this is the ONE file in this repository that is
not byte-identical to its origin. The original module also contains
tournament-session seeding, corpus fixtures, manifest validation, and
CSV/JSON result writers, and imports the private CLI profile builder
(which transitively requires PostgreSQL adapters). The benchmark harness
in this repository uses exactly one function from it: ``bootstrap_ci``.
This slice ships that function alone, byte-identical in body, so the
PostgreSQL/CLI dependency chain never enters the public repository.
See MANIFEST.md for the full origin record.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(
    per_query_deltas: list[float],
    n_resamples: int = 1000,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap 95% CI on mean(delta).

    Returns (ci_lower, ci_upper) where ci_lower is the 2.5th percentile
    and ci_upper the 97.5th percentile of the bootstrap distribution of mean deltas.
    ci_lower > 0 is the primary statistical gate in the ReplacementRule.
    """
    arr = np.array(per_query_deltas, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = arr[indices].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
