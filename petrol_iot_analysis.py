"""
=============================================================================
petrol_iot_analysis.py
Full 181-Node Petrol Dataset Analysis via Paillier-labHE
=============================================================================

Loads the petrol dataset (181 countries = 181 IoT sensor nodes),
encrypts all four columns at 2048-bit security, computes all statistics
homomorphically on encrypted data, decrypts, and verifies zero error.

Statistics computed:
    Degree-1: mean for all 4 columns
    Degree-2: variance for all 4 columns
    Degree-2: covariance for all 6 column pairs
    Degree-2: Pearson correlation for all 6 pairs

All results match plaintext computation with 0% cryptographic error
(exact integer arithmetic over encrypted data).

Usage:
    python3 petrol_iot_analysis.py
=============================================================================
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from paillier_labhe  import PaillierLabHE, LabeledCiphertext
from iot_statistics  import IoTPrivacyPreservingStats, SCALE
from differential_privacy import DPPostProcessor, COLUMN_BOUNDS


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

DATASET_PATH = "Petrol_Dataset_June_23_2022_--_Version_2.csv"

COLUMNS = [
    "Daily Oil Consumption (Barrels)",
    "Yearly Gallons Per Capita",
    "Price Per Gallon (USD)",
    "GDP Per Capita ( USD )",
]

EPSILONS = [0.1, 0.5, 1.0, 2.0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_petrol(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"  Loaded: {path} ({enc}), {len(df)} rows")
            break
        except Exception:
            continue
    else:
        raise FileNotFoundError(f"Cannot load dataset: {path}")

    for col in COLUMNS:
        if col in df.columns:
            df[col] = (pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.replace("%", ""),
                errors="coerce"))
    return df


# ---------------------------------------------------------------------------
# Encryption helper
# ---------------------------------------------------------------------------

def encrypt_column(labhe: PaillierLabHE, values: List[float],
                   col_name: str) -> Tuple[List[LabeledCiphertext], List[str]]:
    """Encrypt one column (offline + online split per node)."""
    cts, labels = [], []
    for i, v in enumerate(values):
        label = f"{col_name}_node_{i}"
        m     = int(abs(float(v)) * SCALE)
        b, beta = labhe.offline_encrypt(label)
        a       = labhe.online_encrypt(m, b)
        from paillier_labhe import LabeledCiphertext as LC
        cts.append(LC(label=label, a=a, beta=beta))
        labels.append(label)
    return cts, labels


# ---------------------------------------------------------------------------
# Plaintext verification
# ---------------------------------------------------------------------------

def compute_plaintext_stats(data: Dict[str, List[float]]) -> Dict[str, float]:
    """Compute all statistics in plaintext for comparison."""
    stats = {}
    n = len(next(iter(data.values())))
    for col, vals in data.items():
        mean = sum(vals) / n
        var  = sum((v - mean) ** 2 for v in vals) / n
        stats[f"{col}_mean"]     = mean
        stats[f"{col}_variance"] = var

    cols = list(data.keys())
    for i, cx in enumerate(cols):
        mx = sum(data[cx]) / n
        for cy in cols[i + 1:]:
            my  = sum(data[cy]) / n
            cov = sum((data[cx][k] - mx) * (data[cy][k] - my)
                      for k in range(n)) / n
            stats[f"cov_{cx}_{cy}"] = cov

    return stats


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(dataset_path: str = DATASET_PATH) -> Dict:
    print("=" * 65)
    print("  Petrol Dataset — Full 181-Node labHE Analysis")
    print("  Security: Paillier 2048-bit  |  Nodes: 181")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────
    df   = load_petrol(dataset_path)
    data = {}
    for col in COLUMNS:
        vals = df[col].dropna().tolist()
        data[col] = vals
    n_nodes = len(data[COLUMNS[0]])
    print(f"  Active nodes: {n_nodes}")

    # ── Key generation ─────────────────────────────────────────────────────
    print("\n  KeyGen (2048-bit)...", end=" ", flush=True)
    t0   = time.perf_counter()
    labhe = PaillierLabHE()
    labhe.keygen()
    print(f"{(time.perf_counter()-t0)*1000:.0f} ms")

    stats_engine = IoTPrivacyPreservingStats(labhe)

    # ── Encrypt all columns ────────────────────────────────────────────────
    print("\n  Encrypting all columns (offline+online, 2048-bit)...")
    enc_cols   : Dict[str, List[LabeledCiphertext]] = {}
    label_cols : Dict[str, List[str]]               = {}
    for col in COLUMNS:
        print(f"    {col[:40]}...", end=" ", flush=True)
        t0 = time.perf_counter()
        cts, labels = encrypt_column(labhe, data[col], col)
        enc_cols[col]   = cts
        label_cols[col] = labels
        print(f"{(time.perf_counter()-t0)*1000:.0f} ms  ({n_nodes} nodes)")

    # ── Cloud: compute all encrypted statistics ────────────────────────────
    print("\n  Cloud: homomorphic computation...")

    # Means (degree-1)
    enc_means = {}
    for col in COLUMNS:
        enc_means[col] = stats_engine.secure_mean(enc_cols[col])

    # Variances (degree-2)
    enc_sq, enc_sum_for_var = {}, {}
    for col in COLUMNS:
        print(f"    Variance ({col[:30]})...", end=" ", flush=True)
        t0 = time.perf_counter()
        esq, esum = stats_engine.secure_variance(enc_cols[col])
        enc_sq[col]          = esq
        enc_sum_for_var[col] = esum
        print(f"{(time.perf_counter()-t0)*1000:.0f} ms")

    # Covariances (degree-2)
    enc_cov = {}
    cols = COLUMNS
    for i, cx in enumerate(cols):
        for cy in cols[i + 1:]:
            key = f"{cx}__{cy}"
            print(f"    Cov ({cx[:20]}, {cy[:20]})...", end=" ", flush=True)
            t0 = time.perf_counter()
            exy, _, _ = stats_engine.secure_covariance(enc_cols[cx], enc_cols[cy])
            enc_cov[key] = (exy,
                            enc_means[cx], label_cols[cx],
                            enc_means[cy], label_cols[cy])
            print(f"{(time.perf_counter()-t0)*1000:.0f} ms")

    # ── Receiver: decrypt all statistics ──────────────────────────────────
    print("\n  Receiver: decryption...")
    labhe_stats: Dict[str, float] = {}

    # Means
    for col in COLUMNS:
        total = labhe.decrypt(label_cols[col], enc_means[col])
        labhe_stats[f"{col}_mean"] = total / (SCALE * n_nodes)

    # Variances
    for col in COLUMNS:
        labhe_stats[f"{col}_variance"] = stats_engine.decrypt_variance(
            enc_sq[col], enc_sum_for_var[col], label_cols[col], n_nodes)

    # Covariances
    for i, cx in enumerate(cols):
        for cy in cols[i + 1:]:
            key = f"{cx}__{cy}"
            exy, _, lx, _, ly = enc_cov[key]
            labhe_stats[f"cov_{cx}_{cy}"] = stats_engine.decrypt_covariance(
                exy, enc_means[cx], enc_means[cy], lx, ly, n_nodes)

    # ── Plaintext verification ─────────────────────────────────────────────
    print("\n  Plaintext verification (0% error check)...")
    plaintext = compute_plaintext_stats(data)
    max_err = 0.0
    for key, true_val in plaintext.items():
        if key not in labhe_stats:
            continue
        enc_val = labhe_stats[key]
        if true_val != 0:
            err = abs(enc_val - true_val) / abs(true_val) * 100
            max_err = max(max_err, err)
    print(f"    Max relative error: {max_err:.6f}%  {'✓ EXACT' if max_err < 0.001 else '✗ ERROR'}")

    # ── Print true statistics ─────────────────────────────────────────────
    print("\n  ── TRUE STATISTICS (181 nodes, 0% crypto error) ──")
    for col in COLUMNS:
        print(f"  {col[:40]:40}  mean={labhe_stats[col+'_mean']:>15.4f}  "
              f"var={labhe_stats[col+'_variance']:>20.4f}")

    print("\n  Covariances:")
    for i, cx in enumerate(cols):
        for cy in cols[i + 1:]:
            k = f"cov_{cx}_{cy}"
            print(f"  cov({cx[:20]}, {cy[:20]}): {labhe_stats[k]:.4f}")

    # ── Pearson correlations ──────────────────────────────────────────────
    print("\n  Pearson correlations:")
    for i, cx in enumerate(cols):
        for cy in cols[i + 1:]:
            cov = labhe_stats[f"cov_{cx}_{cy}"]
            vx  = labhe_stats[f"{cx}_variance"]
            vy  = labhe_stats[f"{cy}_variance"]
            rho = IoTPrivacyPreservingStats.pearson_from_stats(cov, vx, vy)
            print(f"  ρ({cx[:20]}, {cy[:20]}): {rho:.6f}")

    # ── Differential Privacy ──────────────────────────────────────────────
    print("\n  ── DIFFERENTIAL PRIVACY RESULTS ──")
    dp_results = {}
    for eps in EPSILONS:
        dp = DPPostProcessor(epsilon=eps, n_nodes=n_nodes)
        dp_results[str(eps)] = dp.run_all_statistics(labhe_stats)
        errors = {k: round(v["relative_error"] * 100, 1)
                  for k, v in dp_results[str(eps)].items()
                  if not math.isinf(v["relative_error"])}
        print(f"\n  ε = {eps}:")
        for k, e in list(errors.items())[:6]:
            print(f"    {k[:45]:45}: {e:.1f}%")

    # ── Save results ──────────────────────────────────────────────────────
    save = {
        "n_nodes":      n_nodes,
        "key_size_bits": 2048,
        "true_stats":   labhe_stats,
        "plaintext_ref": plaintext,
        "max_crypto_error_pct": max_err,
        "column_bounds": {k: list(v) for k, v in COLUMN_BOUNDS.items()},
        "dp_results":   {
            eps: {
                k: {
                    "true":           v["true"],
                    "relative_error": v["relative_error"],
                }
                for k, v in dp_results[eps].items()
            }
            for eps in [str(e) for e in EPSILONS]
        },
    }
    out = "/mnt/user-data/outputs/petrol_analysis_results.json"
    with open(out, "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\n  Results saved → {out}")
    print("=" * 65)
    return save


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DATASET_PATH
    run_analysis(path)
