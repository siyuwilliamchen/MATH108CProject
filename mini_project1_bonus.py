"""
Mini-Project 1: The Matrix Architect, Bonus Challenge
Math 108C: Applied Linear Algebra
PropTech AI - Design Matrix Builder
"""

import numpy as np
import pandas as pd

# ── Raw Data ──────────────────────────────────────────────────────────────────

RAW_DATA = """ID,SqFt,Type,Condition,Notes
1,2500,House,Good,Standard family home.
2,850,Condo,Fair,Needs new paint.
3,1200,Townhome,Excellent,Fully renovated!
4,3500,Estate,Excellent,Large estate with pool.
5,900,Condo,Poor,Major repairs needed.
6,1100,Loft,Good,Open concept downtown.
7,2800,House,Fair,Old roof good bones.
8,1500,Townhome,Good,Near the park.
9,600,Studio,Excellent,High-rise view.
10,3200,Estate,Good,Gated community.
11,2100,House,Poor,Foreclosure water damage.
12,950,Condo,Good,Low HOA fees.
13,1300,Loft,Fair,Noisy street.
14,1700,Townhome,Excellent,End unit.
15,4000,Mansion,Excellent,Historic landmark.
16,750,Studio,Fair,First floor unit.
17,2300,House,Good,Solar panels included.
18,1400,Bungalow,Poor,Foundation issues.
19,2600,House,Excellent,Chef's kitchen.
20,800,Condo,Good,Walking distance to metro."""

# ── Encoding Schema ───────────────────────────────────────────────────────────

CONDITION_MAP = {
    "Poor":      0,
    "Fair":      1,
    "Good":      2,
    "Excellent": 3,
}

# Fixed alphabetical order — must not change
TYPE_ORDER = ["Bungalow", "Condo", "Estate", "House", "Loft", "Mansion", "Studio", "Townhome"]

# Final column order in X
COLUMN_ORDER = ["SqFt", "ConditionEncoded"] + TYPE_ORDER


# ── Core Function ─────────────────────────────────────────────────────────────

def build_design_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Converts a raw housing DataFrame into a numerical design matrix X.

    Input DataFrame must have columns: ID, SqFt, Type, Condition, Notes
    - ID and Notes are excluded (not features)
    - SqFt is kept as-is (numerical)
    - Condition is ordinal encoded:  Poor=0, Fair=1, Good=2, Excellent=3
    - Type is one-hot encoded in fixed alphabetical order:
      [Bungalow, Condo, Estate, House, Loft, Mansion, Studio, Townhome]

    Returns:
        X: numpy array of shape (n_houses, 10)
        columns: list of column names matching the order in X
    """
    df = df.copy()

    # 1. Ordinal encode Condition
    if not df["Condition"].isin(CONDITION_MAP.keys()).all():
        unknown = df[~df["Condition"].isin(CONDITION_MAP.keys())]["Condition"].unique()
        raise ValueError(f"Unknown Condition value(s): {unknown}")
    df["ConditionEncoded"] = df["Condition"].map(CONDITION_MAP)

    # 2. One-hot encode Type with fixed column order
    if not df["Type"].isin(TYPE_ORDER).all():
        unknown = df[~df["Type"].isin(TYPE_ORDER)]["Type"].unique()
        raise ValueError(f"Unknown Type value(s): {unknown}")
    for t in TYPE_ORDER:
        df[t] = (df["Type"] == t).astype(int)

    # 3. Select and order columns — drop ID, Notes, raw Condition, raw Type
    X_df = df[COLUMN_ORDER]

    # 4. Convert to numpy array
    X = X_df.to_numpy(dtype=float)

    return X, COLUMN_ORDER


# ── Optional Challenge: Merge Condo + Townhome → Attached_Unit ───────────────

TYPE_ORDER_MERGED = ["Bungalow", "Attached_Unit", "Estate", "House", "Loft", "Mansion", "Studio"]
COLUMN_ORDER_MERGED = ["SqFt", "ConditionEncoded"] + TYPE_ORDER_MERGED

def build_design_matrix_merged(df: pd.DataFrame) -> np.ndarray:
    """
    Same as build_design_matrix but merges Condo and Townhome into Attached_Unit.
    Returns X of shape (n_houses, 9).
    """
    df = df.copy()

    # Merge before encoding
    df["Type"] = df["Type"].replace({"Condo": "Attached_Unit", "Townhome": "Attached_Unit"})

    df["ConditionEncoded"] = df["Condition"].map(CONDITION_MAP)

    for t in TYPE_ORDER_MERGED:
        df[t] = (df["Type"] == t).astype(int)

    X_df = df[COLUMN_ORDER_MERGED]
    X = X_df.to_numpy(dtype=float)

    return X, COLUMN_ORDER_MERGED


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import io

    # Load data
    df = pd.read_csv(io.StringIO(RAW_DATA))
    df.columns = df.columns.str.strip()

    print("=" * 60)
    print("BASELINE DESIGN MATRIX")
    print("=" * 60)

    X, cols = build_design_matrix(df)

    print(f"\nColumn order: {cols}")
    print(f"Matrix shape: {X.shape}  (expected: (20, 10))\n")

    # ── Phase 3: Row Audit ────────────────────────────────────────────────────
    # ID=4 is at index 3 (zero-indexed)
    # Raw: SqFt=3500, Type=Estate, Condition=Excellent
    # Expected: [3500, 3, 0, 0, 1, 0, 0, 0, 0, 0]

    print("PHASE 3 — ROW AUDIT (ID=4, index=3)")
    print("-" * 40)
    x4_manual   = [3500, 3, 0, 0, 1, 0, 0, 0, 0, 0]
    x4_computed = X[3]
    print(f"Manual:   {x4_manual}")
    print(f"Computed: {x4_computed.tolist()}")
    match = x4_manual == x4_computed.tolist()
    print(f"Match: {'PASS' if match else 'FAIL'}\n")

    # ── Full Matrix Output ────────────────────────────────────────────────────
    print("FULL DESIGN MATRIX X")
    print("-" * 40)
    X_df_display = pd.DataFrame(X, columns=cols)
    X_df_display.index = df["ID"].values
    X_df_display.index.name = "ID"
    print(X_df_display.to_string())

    # ── Optional Challenge ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OPTIONAL CHALLENGE -- Condo + Townhome -> Attached_Unit")
    print("=" * 60)

    X_merged, cols_merged = build_design_matrix_merged(df)
    print(f"\nColumn order: {cols_merged}")
    print(f"Matrix shape: {X_merged.shape}  (expected: (20, 9))\n")

    X_merged_display = pd.DataFrame(X_merged, columns=cols_merged)
    X_merged_display.index = df["ID"].values
    X_merged_display.index.name = "ID"
    print(X_merged_display.to_string())
