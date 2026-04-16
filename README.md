# paillier-labhe-iot

> **Paper**: *Malicious-Secure Paillier Labeled Homomorphic Encryption
> for Industrial IoT: Degree-2 Statistics, Node Compromise Defence,
> and Differential Privacy*
>
> **Authors**: Chaleu Wansi Hilary · Mohamed-Lamine Messai · Fouotsa Emmanuel
>
> **Affiliations**:
> Centre for Cybersecurity and Mathematical Cryptology, University of Bamenda, Cameroon;
> ERIC, Universite Lumiere Lyon 2 / Universite Claude Bernard Lyon 1, France
>
> **Contact**: hilarychaleu@gmail.com

---

## What this repository contains

Complete Python implementation used to produce every number in the paper.
All benchmark results are **measured at 2048-bit Paillier security**, not
extrapolated. An earlier `key_size=1024` bug has been corrected; all files
here enforce 2048-bit.

The code extends the labHE scheme of Barbosa, Catalano & Fiore (ESORICS 2017)
with five novel contributions:

| ID | Contribution | Proved in | File |
|---|---|---|---|
| C2 | Degree-2 MULT: variance, covariance, Pearson rho | Theorem 1 | `paillier_labhe.py` |
| C3 | 181-node multi-party aggregation, independent keys | Proposition 1 | `paillier_labhe.py` |
| C4 | Epoch key derivation — forward secrecy | Theorem 4 | `paillier_labhe.py` |
| C5 | ZK Sigma range proof — forgery detection | Theorems 2-3 | `paillier_labhe.py` |
| C6 | Pure (epsilon,0)-DP — Laplace mechanism | Section V | `differential_privacy.py` |

---

## Repository structure

```
paillier-labhe-iot/
├── paillier_labhe.py          # Core: Paillier keygen/enc/dec, labHE algorithms 1-5,
│                              #       MULT (Theorem 1), ZK proof (C5), epoch keys (C4),
│                              #       IoTMultiPartyLabHE (181-node scheme)
├── iot_statistics.py          # Encrypted mean, variance, covariance, Pearson rho
├── differential_privacy.py    # Laplace mechanism with exact sensitivity derivations (C6)
├── petrol_iot_analysis.py     # Full 181-node analysis on the petrol dataset
├── benchmark_2048bit.py       # Complete benchmark suite — reproduces Tables I-IV in paper
├── demo.py                    # End-to-end 5-node demonstration (start here)
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

---

## Requirements

```
Python >= 3.8
sympy >= 1.12
cryptography >= 41.0
pandas >= 2.0
numpy >= 1.24
```

Install:
```bash
pip install -r requirements.txt
```

> **gmpy2 is not required.** Python's built-in `pow(a, b, n)` is GMP-backed
> in CPython 3.8+ and achieves equivalent modular exponentiation performance.

---

## Quick start

### Step 1: Run the demonstration

```bash
python3 demo.py
```

Expected final lines:
```
All correctness checks:  PASSED
ZK forgery detection:    PASSED (forged CT rejected)
Forward secrecy (C4):    DEMONSTRATED (K^1 != K^2)
Differential privacy:    APPLIED (epsilon = 1.0)
```

### Step 2: Full 181-node petrol dataset analysis

```bash
python3 petrol_iot_analysis.py /path/to/Petrol_Dataset_June_23_2022_--_Version_2.csv
```

### Step 3: Reproduce benchmark tables (30-60 minutes)

```bash
python3 benchmark_2048bit.py
```

Results written to `benchmark_2048bit_results.json`.

---

## Verified benchmark results (2048-bit, 30 repetitions)

### Core operations (Table I in paper)

| Operation | Mean (ms) | Std (ms) | CI-95 (ms) |
|---|---|---|---|
| Key generation | 611.65 | 384.82 | [259.72, 963.59] |
| Offline Enc / node | 96.796 | 3.232 | [95.59, 98.00] |
| **Online Enc / node** | **0.00058** | 0.00147 | ~0 |
| Hom. addition | 0.04233 | 0.00331 | [0.041, 0.044] |
| Hom. mult. deg-2 | 294.871 | 12.487 | [290.21, 299.53] |
| Online dec. deg-2 | 95.288 | 2.361 | [94.41, 96.17] |
| CT size / node | | | 512 bytes |
| Bandwidth / node | | | 768 bytes |

Online enc speedup: ~166,890x over offline phase.

### 181-node pipeline (Table II in paper, 5 runs)

| Phase | Mean (ms) | Std (ms) |
|---|---|---|
| Offline Enc (181 nodes) | 17,477 | 145 |
| Online Enc (181 nodes) | 0.101 | 0.019 |
| Hom. aggregation | 10.077 | 3.819 |
| **Full online pipeline** | **10.37** | **3.803** |
| Aggregate correctness | True (5/5) | |
| CT payload (181 nodes) | 135.8 KB | |

### Security extensions (100 repetitions, Table I)

| Metric | Value | CI-95 |
|---|---|---|
| ZK prove / node | 0.00422 ms | [0.00331, 0.00513] ms |
| ZK verify / node | 0.00398 ms | [0.00352, 0.00443] ms |
| ZK total 181 nodes | 1.484 ms | |
| Epoch key / call | 0.000847 ms | [0.000752, 0.000941] ms |
| **Combined overhead 181 nodes** | **1.637 ms** | **0.0094% of offline** |
| Forgery rejection rate | 100% | 100 trials |

### Cryptographic accuracy (Table III in paper)

All 12 degree-2 statistics (4 means, 4 variances, 4 covariances) computed over
181 encrypted nodes match plaintext results with **0.000% error**. This is a
structural guarantee of exact Paillier integer arithmetic, not achievable under
approximate-arithmetic schemes (e.g., CKKS).

### Differential privacy (Table IV in paper)

Laplace mechanism, n = 181 nodes:

| Statistic | eps=0.5 | eps=1.0 | eps=2.0 |
|---|---|---|---|
| Daily Oil - Mean | 44.7% | 22.4% | 11.2% |
| Gallons - Mean | 4.8% | 2.4% | 1.2% |
| Price/gal - Mean | 2.9% | 1.5% | 0.7% |
| GDP - Mean | 6.3% | 3.1% | 1.6% |

Recommended: eps = 1.0 (all means < 25%).

---

## Scheme comparison

| Property | Paillier-2048 (this work) | JL13-512 | BFV/SEAL |
|---|---|---|---|
| Security | DCR | 2^k-Res. | Ring-LWE (PQ) |
| Online Enc (ms) | **0.00058** | 0.00018 | ~1091 |
| Dec. deg-2 feasibility | **Any message size** | k<=34 only* | Any |
| CT size (bytes) | **512** | 64 | 32,000-65,000 |
| vs. BFV bandwidth | **64-130x smaller** | ~320x smaller | baseline |
| Forward secrecy (C4) | **YES** | No | No |
| ZK forgery detect. (C5) | **YES** | No | No |
| Differential privacy (C6) | **YES** | Not studied | Not studied |

*JL13 deg-2 decryption infeasible for k>=35 (~32 GB BSGS table required).
This dataset requires k>=35; Paillier has no message-space constraint.

---

## Citation

```bibtex
@inproceedings{wansi2026paillier,
  title     = {Malicious-Secure Paillier Labeled Homomorphic Encryption
               for Industrial IoT: Degree-2 Statistics, Node Compromise
               Defence, and Differential Privacy},
  author    = {Wansi Hilary,Mohamed-Lamine
               and Emmanuel, Fouotsa},
  booktitle = {[Conference name and year]},
  year      = {2026}
}
```

---

## Licence

MIT Licence. Copyright (c) 2026 Chaleu Wansi Hilary.
See the header of each source file for the full licence text.
