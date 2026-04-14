"""
=============================================================================
differential_privacy.py
Differential Privacy Post-Processing for labHE Statistics
=============================================================================

Applies (ε, 0)-DP (Laplace mechanism) to statistics decrypted from
Paillier-labHE encrypted aggregates.

DP operates on the OUTPUT layer — after decryption — to protect against
inference attacks on published aggregate statistics (differencing attacks).
It does NOT protect individual readings from a receiver who holds sk.

Formal guarantee (Dwork & Roth, 2014):
    For any two neighbouring datasets D, D' differing in one node:
        Pr[M(D) ∈ S] ≤ e^ε · Pr[M(D') ∈ S]   for all S

Sensitivity values for our petrol dataset (181 nodes, 4 columns):
    Daily Oil:  range [0, 25,000,000]  →  Δ_mean = 138,122    Δ_var = 3.45×10¹²
    Gallons:    range [0, 2,000]       →  Δ_mean = 11.05       Δ_var = 22,099
    Price:      range [0, 15]          →  Δ_mean = 0.083       Δ_var = 1.24
    GDP:        range [0, 100,000]     →  Δ_mean = 552.5       Δ_var = 5.52×10⁷

Recommended setting: ε = 1.0 (error < 25% for most statistics).
Note: Oil Consumption variance has high error at ε ≤ 1.0 due to extreme
data range — this is a property of the distribution, not a scheme flaw.
=============================================================================
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Laplace mechanism
# ---------------------------------------------------------------------------

def _laplace_sample(scale: float) -> float:
    """Sample from Laplace(0, scale) using inverse CDF."""
    u = random.uniform(-0.5, 0.5)
    return -scale * math.copysign(1, u) * math.log(1 - 2 * abs(u))


def laplace_mechanism(true_value: float, sensitivity: float,
                      epsilon: float) -> float:
    """
    Laplace mechanism for ε-DP.
    Output = true_value + Lap(Δf / ε).

    Args:
        true_value  : decrypted statistic
        sensitivity : global sensitivity Δf
        epsilon     : privacy budget  (smaller = more private)

    Returns:
        Noisy value satisfying ε-DP.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if sensitivity < 0:
        raise ValueError("sensitivity must be ≥ 0")
    if sensitivity == 0:
        return true_value
    scale = sensitivity / epsilon
    return true_value + _laplace_sample(scale)


# ---------------------------------------------------------------------------
# Sensitivity formulas (exact, for our 4-column petrol dataset)
# ---------------------------------------------------------------------------

def sensitivity_mean(lo: float, hi: float, n: int) -> float:
    """Δ(mean) = (hi − lo) / n."""
    return (hi - lo) / n


def sensitivity_variance(lo: float, hi: float, n: int) -> float:
    """Δ(variance) = (hi − lo)² / n  [upper bound, Lemma 1]."""
    return (hi - lo) ** 2 / n


def sensitivity_covariance(lo_x: float, hi_x: float,
                            lo_y: float, hi_y: float, n: int) -> float:
    """Δ(cov) = (hi_x − lo_x)(hi_y − lo_y) / n."""
    return (hi_x - lo_x) * (hi_y - lo_y) / n


# ---------------------------------------------------------------------------
# Column bounds (petrol dataset, 181 nodes)
# ---------------------------------------------------------------------------

COLUMN_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Daily Oil Consumption (Barrels)": (0.0, 25_000_000.0),
    "Yearly Gallons Per Capita":       (0.0, 2_000.0),
    "Price Per Gallon (USD)":          (0.0, 15.0),
    "GDP Per Capita ( USD )":          (0.0, 100_000.0),
}

N_NODES = 181


# ---------------------------------------------------------------------------
# DPPostProcessor — high-level API
# ---------------------------------------------------------------------------

class DPPostProcessor:
    """
    Post-process decrypted labHE statistics with (ε, 0)-DP.

    Workflow:
        1. labHE computes statistics over encrypted data (cloud)
        2. Receiver decrypts → true plaintext statistics
        3. DPPostProcessor adds calibrated Laplace noise  (this class)
        4. Publish noisy statistics satisfying ε-DP
    """

    def __init__(self, epsilon: float, n_nodes: int = N_NODES,
                 seed: Optional[int] = None):
        """
        Args:
            epsilon  : privacy budget (0 < ε; typical: 0.1, 0.5, 1.0, 2.0)
            n_nodes  : number of sensor nodes (181 for petrol dataset)
            seed     : optional random seed for reproducibility
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        self.epsilon  = epsilon
        self.n_nodes  = n_nodes
        if seed is not None:
            random.seed(seed)

    def privatize_mean(self, true_mean: float, column: str) -> Tuple[float, float]:
        """
        Add ε-DP noise to a mean statistic.

        Returns:
            (noisy_mean, sensitivity_used)
        """
        lo, hi = COLUMN_BOUNDS[column]
        sens   = sensitivity_mean(lo, hi, self.n_nodes)
        noisy  = laplace_mechanism(true_mean, sens, self.epsilon)
        return noisy, sens

    def privatize_variance(self, true_var: float, column: str) -> Tuple[float, float]:
        """Add ε-DP noise to a variance statistic."""
        lo, hi = COLUMN_BOUNDS[column]
        sens   = sensitivity_variance(lo, hi, self.n_nodes)
        noisy  = laplace_mechanism(true_var, sens, self.epsilon)
        return noisy, sens

    def privatize_covariance(self, true_cov: float,
                             col_x: str, col_y: str) -> Tuple[float, float]:
        """Add ε-DP noise to a covariance statistic."""
        lo_x, hi_x = COLUMN_BOUNDS[col_x]
        lo_y, hi_y = COLUMN_BOUNDS[col_y]
        sens = sensitivity_covariance(lo_x, hi_x, lo_y, hi_y, self.n_nodes)
        noisy = laplace_mechanism(true_cov, sens, self.epsilon)
        return noisy, sens

    def relative_error(self, true_val: float, noisy_val: float) -> float:
        """Relative error |noisy − true| / |true| (returns inf if true = 0)."""
        if true_val == 0:
            return float("inf")
        return abs(noisy_val - true_val) / abs(true_val)

    def privacy_guarantee(self) -> str:
        """Human-readable DP guarantee."""
        return (f"ε = {self.epsilon}: each output changes by at most "
                f"exp({self.epsilon}) = {math.exp(self.epsilon):.3f} "
                f"in probability when one node's data changes.")

    def run_all_statistics(
        self, true_stats: Dict[str, float]
    ) -> Dict[str, Dict]:
        """
        Privatize all statistics in true_stats dict.

        Expected keys (matching petrol dataset):
            "<column>_mean", "<column>_variance",
            "cov_<col_x>_<col_y>"

        Returns:
            Dict mapping stat_key → {
                "true": float,
                "noisy": float,
                "sensitivity": float,
                "relative_error": float,
            }
        """
        results = {}
        cols = list(COLUMN_BOUNDS.keys())

        for col in cols:
            key_mean = f"{col}_mean"
            if key_mean in true_stats:
                noisy, sens = self.privatize_mean(true_stats[key_mean], col)
                results[key_mean] = {
                    "true": true_stats[key_mean],
                    "noisy": noisy,
                    "sensitivity": sens,
                    "relative_error": self.relative_error(true_stats[key_mean], noisy),
                }

            key_var = f"{col}_variance"
            if key_var in true_stats:
                noisy, sens = self.privatize_variance(true_stats[key_var], col)
                results[key_var] = {
                    "true": true_stats[key_var],
                    "noisy": noisy,
                    "sensitivity": sens,
                    "relative_error": self.relative_error(true_stats[key_var], noisy),
                }

        # Covariances
        for i, cx in enumerate(cols):
            for cy in cols[i + 1:]:
                key_cov = f"cov_{cx}_{cy}"
                if key_cov in true_stats:
                    noisy, sens = self.privatize_covariance(
                        true_stats[key_cov], cx, cy)
                    results[key_cov] = {
                        "true": true_stats[key_cov],
                        "noisy": noisy,
                        "sensitivity": sens,
                        "relative_error": self.relative_error(
                            true_stats[key_cov], noisy),
                    }
        return results
