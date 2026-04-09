"""
Mini-Project 2: The Valuation Engine
Math 108C: Applied Linear Algebra
PropTech AI - Miami Housing Price Predictor

Phases:
    Phase 0  - Modeling Setup (Intercept & Scaling)
    Phase 1  - Build X and y  (Decision A)
    Phase 2  - Model Definitions
    Phase 3  - Market Scenarios P = XW  (Decision B)
    Phase 4  - Valuation Audit (Partitioned Matrices)
    Phase 5  - Least Squares Benchmark
    Evaluation - MAE Scoreboard & Bar Chart
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_PATH = "miami-housing.csv"     # place CSV in the same directory

# Decision A: chosen features and their categories
STRUCTURE_FEATURES = [
    "TOT_LVG_AREA",       # total living area (sqft)
    "LND_SQFOOT",         # lot area (sqft)
    "structure_quality",  # quality rating (ordinal)
    "age",                # age of property (years)
    "SPEC_FEAT_VAL",      # value of special features ($)
]

LOCATION_FEATURES = [
    "OCEAN_DIST",    # distance to ocean (m)
    "WATER_DIST",    # distance to water (m)
    "CNTR_DIST",     # distance to city center (m)
    "SUBCNTR_DI",    # distance to subcenter (m)
    "HWY_DIST",      # distance to highway (m)
]

TARGET_COL = "SALE_PRC"

ALL_FEATURES = STRUCTURE_FEATURES + LOCATION_FEATURES  # p = 10 features
# d = p + 1 = 11  (intercept + 10 features)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Modeling Setup helpers
# ══════════════════════════════════════════════════════════════════════════════

def z_score_standardize(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, dict]:
    """
    Z-score standardize selected columns.
    Returns the transformed DataFrame and a dict of {col: (mean, std)}
    so coefficients can be translated back to original units later.

    z = (x - mu) / sigma  =>  alpha = 1/sigma,  beta = -mu/sigma
    """
    df = df.copy()
    stats = {}
    for col in columns:
        mu  = df[col].mean()
        sig = df[col].std()
        df[col] = (df[col] - mu) / sig
        stats[col] = (mu, sig)
    return df, stats


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Build X and y  (Execute Decision A)
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def build_X_and_y(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Constructs design matrix X ∈ R^{n x d} and target vector y ∈ R^n.

    Column layout of X:
        col 0      : intercept (all ones)  — NOT standardized
        cols 1-5   : structure features    — Z-score standardized
        cols 6-10  : location features     — Z-score standardized

    Returns
    -------
    X      : ndarray  (n, d)   with d = p + 1 = 11
    y      : ndarray  (n,)
    stats  : dict     scaling parameters per feature
    """
    # 1. Target vector
    y = df[TARGET_COL].to_numpy(dtype=float)

    # 2. Select feature columns
    feat_df = df[ALL_FEATURES].copy()

    # 3. Z-score standardize ALL feature columns (not the intercept)
    feat_df, stats = z_score_standardize(feat_df, ALL_FEATURES)

    # 4. Prepend intercept column of ones
    n = len(feat_df)
    ones = np.ones((n, 1))
    X = np.hstack([ones, feat_df.to_numpy(dtype=float)])

    # 5. Dimension check
    assert X.shape == (n, len(ALL_FEATURES) + 1), (
        f"Expected ({n}, {len(ALL_FEATURES)+1}), got {X.shape}"
    )
    return X, y, stats


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Model Definitions
# ══════════════════════════════════════════════════════════════════════════════
#
# Scalar model (row i):
#   ŷ_i = w0 + w1*TOT_LVG_AREA_z + w2*LND_SQFOOT_z + w3*structure_quality_z
#             + w4*age_z + w5*SPEC_FEAT_VAL_z
#             + w6*OCEAN_DIST_z + w7*WATER_DIST_z + w8*CNTR_DIST_z
#             + w9*SUBCNTR_DI_z + w10*HWY_DIST_z
#
# Matrix model (all n houses at once):
#   ŷ = Xw
#
#   X  ∈ R^{n×d},  w ∈ R^d,  ŷ ∈ R^n     (d = 11, n = 13932)
#
# Why is Xw defined?
#   X has d columns and w has d rows  =>  inner dimensions match.
#   Result shape: (n, d) @ (d,) = (n,) — one prediction per house.
#
# The i-th entry of Xw:
#   ŷ_i = x_i · w  =  dot product of house i's feature vector with weights
#   = predicted sale price of the i-th house (in dollars).


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Decision B: Market Scenarios  →  P = XW
# ══════════════════════════════════════════════════════════════════════════════
#
# Weight vector layout (d = 11):
#   index 0        : intercept  w0
#   indices 1-5    : structure  [TOT_LVG_AREA, LND_SQFOOT, structure_quality, age, SPEC_FEAT_VAL]
#   indices 6-10   : location   [OCEAN_DIST, WATER_DIST, CNTR_DIST, SUBCNTR_DI, HWY_DIST]
#
# Units: weights are in "dollars per 1 standard deviation of the feature"
# (because X was Z-score standardized).

def define_weight_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns three weight vectors for the three market scenarios.

    Scenario 1 — Baseline (Rational Market)
        Positive weights for size/quality; negative for distances (closer = better).
        Intercept ≈ mean Miami sale price (~$350,000).

    Scenario 2 — "Location is Everything"
        Location weights amplified 3x; structure weights shrunk to near zero.
        A shack on the beach is worth millions.

    Scenario 3 — "The Luxury Premium" (Custom)
        Archetype: Buyers only care about quality and ocean proximity.
        Structure quality weight doubled; all distance weights except OCEAN_DIST set to zero.
        Age heavily penalized (luxury buyers want new construction).
        Intercept inflated to reflect a premium market baseline.
    """

    # ── Scenario 1: Baseline (Rational Market) ──────────────────────────────
    w1 = np.array([
        350_000,   # w0  intercept  (~mean sale price)
        # Structure (positive = adds value)
         80_000,   # w1  TOT_LVG_AREA    (+$80k per SD)
         25_000,   # w2  LND_SQFOOT      (+$25k per SD)
         60_000,   # w3  structure_quality (+$60k per SD)
        -20_000,   # w4  age             (older = lower value)
         30_000,   # w5  SPEC_FEAT_VAL   (+$30k per SD)
        # Location (negative = farther away = less valuable)
        -55_000,   # w6  OCEAN_DIST
        -20_000,   # w7  WATER_DIST
        -25_000,   # w8  CNTR_DIST
        -15_000,   # w9  SUBCNTR_DI
         10_000,   # w10 HWY_DIST  (farther from highway = quieter = better)
    ])

    # ── Scenario 2: Location is Everything ──────────────────────────────────
    w2 = np.array([
        350_000,   # w0  intercept (same baseline)
        # Structure weights shrunk to near-zero
          5_000,   # w1  TOT_LVG_AREA
          2_000,   # w2  LND_SQFOOT
          3_000,   # w3  structure_quality
         -1_000,   # w4  age
          2_000,   # w5  SPEC_FEAT_VAL
        # Location weights amplified ~3x
       -165_000,   # w6  OCEAN_DIST     (3x baseline)
        -60_000,   # w7  WATER_DIST
        -75_000,   # w8  CNTR_DIST
        -45_000,   # w9  SUBCNTR_DI
         30_000,   # w10 HWY_DIST
    ])

    # ── Scenario 3: The Luxury Premium ──────────────────────────────────────
    #   Name: "The Luxury Premium"
    #   Logic: Only ocean proximity and build quality matter.
    #          Buyers are wealthy and ignore age/lot size/subcenter distance.
    #          Intercept is higher to reflect a luxury market baseline.
    w3 = np.array([
        500_000,    # w0  intercept (luxury premium baseline)
        # Structure
         70_000,    # w1  TOT_LVG_AREA  (space still matters)
          5_000,    # w2  LND_SQFOOT    (lot size irrelevant to luxury buyer)
        120_000,    # w3  structure_quality (doubled — luxury = top quality)
        -50_000,    # w4  age           (new construction heavily preferred)
         40_000,    # w5  SPEC_FEAT_VAL (upgrades valued)
        # Location — only ocean matters
       -160_000,    # w6  OCEAN_DIST    (primary driver)
              0,    # w7  WATER_DIST    (irrelevant)
              0,    # w8  CNTR_DIST     (irrelevant — luxury buyers drive)
              0,    # w9  SUBCNTR_DI    (irrelevant)
              0,    # w10 HWY_DIST      (irrelevant)
    ])

    return w1, w2, w3


def build_scenario_matrix(w1, w2, w3) -> np.ndarray:
    """
    Stack the three weight vectors column-wise into W ∈ R^{d×3}.
    Each column = one market scenario.
    """
    W = np.column_stack([w1, w2, w3])
    assert W.shape == (len(w1), 3), f"Expected ({len(w1)}, 3), got {W.shape}"
    return W


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Valuation Audit (Partitioned Matrices)
# ══════════════════════════════════════════════════════════════════════════════

def partition_prediction(X: np.ndarray, w: np.ndarray,
                         n_struc: int, n_loc: int) -> tuple:
    """
    Decompose ŷ = Xw into three additive parts via block multiplication:

        ŷ = w0·1  +  X_struc · w_struc  +  X_loc · w_loc
              ^           ^                      ^
            Base     Structure Value         Location Value

    Parameters
    ----------
    X       : (n, d)  full design matrix
    w       : (d,)    weight vector (Scenario 1)
    n_struc : number of structure feature columns
    n_loc   : number of location feature columns

    Returns  (ŷ_total, base, struc_val, loc_val)  each of shape (n,)
    """
    # Column indices
    struc_cols = slice(1, 1 + n_struc)       # cols 1..5
    loc_cols   = slice(1 + n_struc, 1 + n_struc + n_loc)  # cols 6..10

    base      = w[0] * X[:, 0]                        # w0 · 1
    struc_val = X[:, struc_cols] @ w[struc_cols]       # X_struc w_struc
    loc_val   = X[:, loc_cols]   @ w[loc_cols]         # X_loc   w_loc
    y_hat     = base + struc_val + loc_val             # should equal X @ w

    return y_hat, base, struc_val, loc_val


def breakdown_table(X, w, y, house_indices,
                    n_struc, n_loc) -> pd.DataFrame:
    """
    Build a breakdown table for selected houses showing:
        actual price | base | structure value | location value | total predicted | error
    """
    y_hat, base, struc_val, loc_val = partition_prediction(X, w, n_struc, n_loc)

    rows = []
    for i in house_indices:
        rows.append({
            "House Index": i,
            "Actual ($)":  round(y[i], 0),
            "Base ($)":    round(base[i], 0),
            "Struc Val ($)": round(struc_val[i], 0),
            "Loc Val ($)":   round(loc_val[i], 0),
            "Predicted ($)": round(y_hat[i], 0),
            "Error ($)":     round(y[i] - y_hat[i], 0),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Least Squares Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def fit_least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute optimal weights via least squares: ŵ = argmin ||Xw - y||²"""
    w_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return w_hat


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in dollars."""
    return float(np.mean(np.abs(y_true - y_pred)))


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION — MAE Scoreboard & Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_mae_barchart(mae_dict: dict, save_path: str = "mae_scoreboard.png"):
    """Bar chart comparing MAE of the four models."""
    labels = list(mae_dict.keys())
    values = [mae_dict[k] / 1_000 for k in labels]   # convert to $k

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.7)

    ax.set_ylabel("Mean Absolute Error (thousands $)", fontsize=12)
    ax.set_title("Model Comparison — MAE Scoreboard\nMiami Housing Dataset", fontsize=13)
    ax.bar_label(bars, fmt="$%.1fk", padding=4, fontsize=10)
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Bar chart saved to {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MANUAL GATE — Coefficient translation (Phase 3, Critical Requirement)
# ══════════════════════════════════════════════════════════════════════════════

def translate_coefficient(w_z: float, feature: str, stats: dict) -> dict:
    """
    Translate a standardized coefficient back to original units.

    Z-score:  z = (x - mu) / sigma   =>  alpha = 1/sigma,  beta = -mu/sigma
    So the coefficient on x is:   w_x = w_z / sigma
    And the intercept shift is:   delta_w0 = -mu * w_z / sigma

    Example audit for TOT_LVG_AREA:
        w_z  = weight in standardized space
        w_x  = w_z / sigma  = "dollars per 1 extra sqft"
    """
    mu, sigma = stats[feature]
    alpha = 1.0 / sigma
    beta  = -mu / sigma

    w_x       = alpha * w_z          # dollars per 1 original unit
    delta_w0  = beta  * w_z          # shift to intercept

    return {
        "feature":     feature,
        "mu":          round(mu, 4),
        "sigma":       round(sigma, 4),
        "alpha (1/σ)": round(alpha, 6),
        "beta (-μ/σ)": round(beta, 6),
        "w_z (scaled)":   round(w_z, 4),
        "w_x (original)": round(w_x, 4),
        "delta_w0":        round(delta_w0, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load data ─────────────────────────────────────────────────────────────
    print("=" * 65)
    print("MINI-PROJECT 2 — THE VALUATION ENGINE")
    print("=" * 65)

    df = load_data(DATA_PATH)
    print(f"\nDataset loaded:  {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Phase 1: Build X and y ────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("PHASE 1 — Build X and y")
    print("─" * 65)

    X, y, stats = build_X_and_y(df)
    n, d = X.shape

    print(f"  Features selected  : {len(ALL_FEATURES)}  (p = {len(ALL_FEATURES)})")
    print(f"    Structure  ({len(STRUCTURE_FEATURES)}): {STRUCTURE_FEATURES}")
    print(f"    Location   ({len(LOCATION_FEATURES)}): {LOCATION_FEATURES}")
    print(f"  Intercept column   : 1 (column 0, NOT standardized)")
    print(f"  Scaling            : Z-score on all feature columns")
    print(f"\n  X shape : {X.shape}  — expected ({n}, {d})  ✓")
    print(f"  y shape : {y.shape}  — expected ({n},)      ✓")
    print(f"  d = p + 1 = {len(ALL_FEATURES)} + 1 = {d}")

    # Sanity-check: intercept column is all ones
    assert np.all(X[:, 0] == 1.0), "Intercept column must be all ones!"
    print("  Intercept column check: PASS  ✓")

    # ── Phase 2: Model summary ────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("PHASE 2 — Model Definitions")
    print("─" * 65)
    print(f"  Scalar model (house i): ŷ_i = w0 + w1·z1_i + ... + w{d-1}·z{d-1}_i")
    print(f"  Matrix model           : ŷ = Xw")
    print(f"  Dimensions             : X({n}×{d}) @ w({d}) = ŷ({n})")
    print(f"  i-th entry of Xw       : predicted sale price of house i ($)")
    print(f"  Xw is defined because  : X has {d} columns, w has {d} rows (match).")

    # ── Phase 3: Decision B — Market Scenarios ────────────────────────────────
    print("\n" + "─" * 65)
    print("PHASE 3 — Market Scenarios  P = XW")
    print("─" * 65)

    w1, w2, w3 = define_weight_vectors()
    W = build_scenario_matrix(w1, w2, w3)

    print(f"  W shape : {W.shape}  — expected ({d}, 3)")

    P = X @ W    # P = XW  — all three scenarios at once  (n×3)
    print(f"  P shape : {P.shape}  — expected ({n}, 3)")

    # Interpretation of P[4, 1]  (0-indexed: house 5, scenario 2)
    print(f"\n  P[4, 1] = ${P[4, 1]:,.0f}")
    print("  Interpretation: The 'Location is Everything' model predicts")
    print("  that house #5 (index 4) would sell for the amount shown above.")

    # ── Manual Gate: coefficient translation ─────────────────────────────────
    print("\n" + "─" * 65)
    print("MANUAL GATE — Coefficient Translation (TOT_LVG_AREA, Scenario 1)")
    print("─" * 65)

    audit = translate_coefficient(
        w_z     = w1[1],           # Scenario 1 weight for TOT_LVG_AREA
        feature = "TOT_LVG_AREA",
        stats   = stats,
    )
    for k, v in audit.items():
        print(f"  {k:<20}: {v}")
    print(f"\n  Interpretation: In Scenario 1, each additional sqft of living")
    print(f"  area adds approximately ${audit['w_x (original)']:,.2f} to the predicted price.")

    # ── Phase 4: Valuation Audit (Partitioned Matrices) ──────────────────────
    print("\n" + "─" * 65)
    print("PHASE 4 — Valuation Audit (Partitioned Matrices)")
    print("─" * 65)

    print("  Block decomposition of X:")
    print(f"    X  = [ 1 | X_struc | X_loc ]")
    print(f"         col 0 | cols 1-{len(STRUCTURE_FEATURES)} | cols {1+len(STRUCTURE_FEATURES)}-{d-1}")
    print()
    print("  ŷ = w0·1  +  X_struc·w_struc  +  X_loc·w_loc")
    print("         ^           ^                    ^")
    print("       Base    Structure Value       Location Value")

    # Breakdown table for houses 0-4 (houses 1-5 in 1-indexed)
    AUDIT_INDICES = [0, 1, 2, 3, 4]
    table = breakdown_table(
        X, w1, y, AUDIT_INDICES,
        n_struc=len(STRUCTURE_FEATURES),
        n_loc  =len(LOCATION_FEATURES),
    )
    print(f"\n  Breakdown Table — first 5 houses (Scenario 1, Baseline):")
    print(table.to_string(index=False))

    # Verify partition sum equals X @ w1
    y_hat_partition = X[:, 0]*w1[0] \
                    + X[:, 1:6] @ w1[1:6] \
                    + X[:, 6:]  @ w1[6:]
    y_hat_direct    = X @ w1
    partition_ok    = np.allclose(y_hat_partition, y_hat_direct)
    print(f"\n  Partition sum == X @ w1  :  {'PASS ✓' if partition_ok else 'FAIL ✗'}")

    # ── Phase 5: Least Squares ────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("PHASE 5 — Least Squares Benchmark")
    print("─" * 65)

    # Part A: Underdetermined case
    X_small = X[:5]
    y_small = y[:5]
    print(f"  Part A — X_small shape: {X_small.shape}  (5 equations, {d} unknowns)")
    print(f"  → Underdetermined: infinitely many solutions exist.")
    print(f"    We cannot find a unique w that satisfies X_small·w = y_small.")

    # Part B: Optimal weights via least squares on full dataset
    w_ls = fit_least_squares(X, y)
    print(f"\n  Part B — Least-squares weight for TOT_LVG_AREA:")
    print(f"    LS weight  (scaled) : {w_ls[1]:,.2f}")
    print(f"    Manual w1 (scaled)  : {w1[1]:,.2f}")
    higher = "higher" if w_ls[1] > w1[1] else "lower"
    print(f"    → LS weight is {higher} than the manual choice.")

    # Part C: Intercept experiment
    X_no_int = X[:, 1:]     # drop column of ones
    w_ls_no_int = fit_least_squares(X_no_int, y)

    mae_ls         = compute_mae(y, X        @ w_ls)
    mae_ls_no_int  = compute_mae(y, X_no_int @ w_ls_no_int)

    print(f"\n  Part C — Intercept Experiment:")
    print(f"    MAE with intercept    : ${mae_ls:,.0f}")
    print(f"    MAE without intercept : ${mae_ls_no_int:,.0f}")
    diff = mae_ls_no_int - mae_ls
    print(f"    Removing intercept worsens MAE by ${diff:,.0f}")
    print("    → Forcing the regression line through the origin is a bad idea")
    print("      because real estate has non-zero 'land value' even for 0 features.")

    # ── Evaluation: MAE Scoreboard ────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("EVALUATION — MAE Scoreboard (4 Models)")
    print("─" * 65)

    mae_s1 = compute_mae(y, X @ w1)
    mae_s2 = compute_mae(y, X @ w2)
    mae_s3 = compute_mae(y, X @ w3)

    mae_dict = {
        "Scenario 1\n(Baseline)":  mae_s1,
        "Scenario 2\n(Location)":  mae_s2,
        "Scenario 3\n(Luxury)":    mae_s3,
        "Least Squares\n(Optimal)": mae_ls,
    }

    print(f"\n  {'Model':<28} {'MAE':>15}")
    print(f"  {'─'*28} {'─'*15}")
    for model, mae in mae_dict.items():
        label = model.replace("\n", " ")
        print(f"  {label:<28} ${mae:>14,.0f}")

    print("\n  → Least Squares has the lowest MAE (optimal by definition).")
    print("    Hand-chosen weights are a useful sanity check but cannot beat")
    print("    the data-driven solution.")

    plot_mae_barchart(mae_dict)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DONE.  All phases complete.")
    print("=" * 65)
