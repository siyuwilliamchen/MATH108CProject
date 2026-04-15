# Mini-Project 2: The Valuation Engine — Deliverables
**Math 108C: Applied Linear Algebra | PropTech AI — Miami Housing Price Predictor**

---

## 1. Phase 0: Modeling Setup (Intercept & Scaling)

### Intercept Concept Check

**Q: If you enforce w₀ = 0 (no intercept), what does your model imply about the "Fixed Costs" of owning land?**

Forcing w₀ = 0 implies that a property with zero feature values (zero square footage, no quality, no special features, etc.) has a predicted price of $0. This is economically nonsensical — even a bare plot of land carries fixed costs such as property taxes, utility hookup rights, and zoning entitlements. The intercept captures this irreducible baseline value that exists independent of any measurable feature.

**Q: Is the intercept w₀ a "Property Feature" or a "Market Feature"?**

The intercept is a **Market Feature**. It does not describe any physical attribute of a specific house; rather, it reflects the baseline level of prices set by broad market conditions — the minimum a buyer must pay to participate in the Miami housing market at all, before any property characteristics are considered.

---

### Scaling Concept Check

**Setup:** Given a scaled predictor z = αx + β, with model ŷ = w₀ + w_z · z.

**1. Rewrite in terms of original variable x:**

Substituting z = αx + β:

ŷ = w₀ + w_z(αx + β) = (w₀ + βw_z) + (αw_z)x

So:
- **w̃₀ = w₀ + βw_z**
- **w̃_x = αw_z**

**2. Unit conversion case (z = cx):** Here α = c, β = 0, so w̃_x = c · w_z. The coefficient w_z means "dollars per 1 unit of z" (e.g., dollars per square meter), while w̃_x means "dollars per 1 unit of x" (dollars per square foot). They differ by the conversion factor c.

**3. Standardization case (z = (x − μ)/σ):** Here α = 1/σ and β = −μ/σ, so:
- w̃_x = w_z / σ  (dollars per 1 original unit of x)
- w̃₀ = w₀ − (μ/σ) · w_z

**4. Project decision — scaling applied:**

All 10 feature columns were Z-score standardized. The intercept column (all ones) was left unstandardized. For each feature, the scaling parameters are:

| Feature | Scaling | α = 1/σ | β = −μ/σ |
|---|---|---|---|
| TOT_LVG_AREA | Z-score | 0.001229 | −2.529744 |
| LND_SQFOOT | Z-score | (1/σ) | (−μ/σ) |
| structure_quality | Z-score | (1/σ) | (−μ/σ) |
| age | Z-score | (1/σ) | (−μ/σ) |
| SPEC_FEAT_VAL | Z-score | (1/σ) | (−μ/σ) |
| OCEAN_DIST | Z-score | (1/σ) | (−μ/σ) |
| WATER_DIST | Z-score | (1/σ) | (−μ/σ) |
| CNTR_DIST | Z-score | (1/σ) | (−μ/σ) |
| SUBCNTR_DI | Z-score | (1/σ) | (−μ/σ) |
| HWY_DIST | Z-score | (1/σ) | (−μ/σ) |

---

## 2. Decision Point A: Matrix Blueprint

### Theory Checkpoint

**1. Why is Xw defined, and what is the size of Xw?**

X ∈ ℝ^(n×d) has d columns, and w ∈ ℝ^d has d rows. Because the inner dimensions match (d = d), the matrix-vector product Xw is defined. The result has the outer dimensions, giving a vector of size n×1 — one prediction per house.

**2. In the housing context, what does the i-th entry of Xw represent?**

The i-th entry of Xw is the **predicted sale price of the i-th house** (in dollars), computed as the dot product of that house's feature vector x_i with the weight vector w: ŷᵢ = xᵢ · w = w₀ + w₁z₁ᵢ + ⋯ + w₁₀z₁₀ᵢ.

**3. Why is XW defined, and what is the size of XW?**

X ∈ ℝ^(n×d) and W ∈ ℝ^(d×3). The inner dimensions both equal d, so the product is defined. The result P = XW ∈ ℝ^(n×3) — n rows (one per house) and 3 columns (one per market scenario).

**4. What does the entry P₅,₂ of P = XW mean?**

P₅,₂ is the **predicted sale price of house #5 (1-indexed) under Scenario 2** ("Location is Everything"). It is the price the location-weighted model assigns to the fifth house. Numerically (0-indexed as P[4,1]): **$720,832.92**.

**5. Explain why Xw = w₀·1 + X_struc·w_struc + X_loc·w_loc:**

Because X is partitioned as [**1** | X_struc | X_loc] and w is partitioned correspondingly as [w₀; w_struc; w_loc], block matrix multiplication distributes the product column-block by row-block:

Xw = **1**·w₀ + X_struc·w_struc + X_loc·w_loc

Each block multiplies only its matching weight sub-vector, and the results add linearly — exactly the block multiplication rule from Chapter 2.

**6. Why would WX not make sense as the prediction matrix?**

W ∈ ℝ^(d×3) and X ∈ ℝ^(n×d). For WX to be defined, the inner dimensions would need to match: W has 3 columns but X has n = 13,932 rows. Since 3 ≠ 13,932, the product WX is not defined. Even if dimensions were forced, the result would not give one prediction per house per scenario — it would have the wrong semantic meaning entirely.

---

### Chosen Features

**p = 10 features total | d = p + 1 = 11 (with intercept)**

| # | Feature | Category | Justification |
|---|---|---|---|
| 1 | TOT_LVG_AREA | Structure | Primary size driver; larger living area → higher price |
| 2 | LND_SQFOOT | Structure | Lot size affects land value |
| 3 | structure_quality | Structure | Ordinal quality rating directly affects desirability |
| 4 | age | Structure | Older properties depreciate; newer commands premium |
| 5 | SPEC_FEAT_VAL | Structure | Upgrades (pool, renovations) add measurable value |
| 6 | OCEAN_DIST | Location | Distance to ocean; strongest location price driver in Miami |
| 7 | WATER_DIST | Location | Distance to any water body |
| 8 | CNTR_DIST | Location | Distance to city center; affects urban convenience |
| 9 | SUBCNTR_DI | Location | Distance to subcenter; secondary accessibility factor |
| 10 | HWY_DIST | Location | Distance to highway; farther = quieter = premium |

**Final dimensions:**
- X ∈ ℝ^(13932 × 11)
- w ∈ ℝ^11
- ŷ ∈ ℝ^13932

**Column layout of X:**

| Column | Content |
|---|---|
| 0 | Intercept (all ones, not standardized) |
| 1–5 | Structure features (Z-score standardized) |
| 6–10 | Location features (Z-score standardized) |

---

## 3. Phase 2: Model Definitions

### Scalar Model (single house i)

ŷᵢ = w₀ + w₁·TOT_LVG_AREA_z,i + w₂·LND_SQFOOT_z,i + w₃·structure_quality_z,i + w₄·age_z,i + w₅·SPEC_FEAT_VAL_z,i + w₆·OCEAN_DIST_z,i + w₇·WATER_DIST_z,i + w₈·CNTR_DIST_z,i + w₉·SUBCNTR_DI_z,i + w₁₀·HWY_DIST_z,i

### Matrix Model (all n houses simultaneously)

**ŷ = Xw**

- ŷ ∈ ℝ^13932 — predicted sale prices
- X ∈ ℝ^(13932×11) — design matrix
- w ∈ ℝ^11 — weight vector

**Why Xw is defined:** X has 11 columns and w has 11 rows. The inner dimensions match (11 = 11), so the product is defined, yielding a vector of shape (13932,).

**What the i-th entry of Xw represents:** It is the predicted sale price (in dollars) of the i-th house — the dot product of house i's standardized feature vector with the weight vector w.

---

## 4. Decision Point B: Market Scenarios

### Scenario 3 — "The Luxury Premium"

**Name:** The Luxury Premium

**Logic:** This scenario models a high-end buyer segment (e.g., second-home buyers, international investors) who care only about two things: build quality and ocean proximity. Lot size, distance to subcenters, water features, and highway access are irrelevant to this buyer. New construction is strongly preferred, so age is heavily penalized. The baseline intercept is inflated to $500,000 to reflect a premium market entry point.

### Weight Vectors

```
Index  Feature            w1 (Baseline)   w2 (Location)   w3 (Luxury)
  0    Intercept          $350,000        $350,000        $500,000
  1    TOT_LVG_AREA       +$80,000        +$5,000         +$70,000
  2    LND_SQFOOT         +$25,000        +$2,000         +$5,000
  3    structure_quality  +$60,000        +$3,000         +$120,000
  4    age                -$20,000        -$1,000         -$50,000
  5    SPEC_FEAT_VAL      +$30,000        +$2,000         +$40,000
  6    OCEAN_DIST         -$55,000        -$165,000       -$160,000
  7    WATER_DIST         -$20,000        -$60,000        $0
  8    CNTR_DIST          -$25,000        -$75,000        $0
  9    SUBCNTR_DI         -$15,000        -$45,000        $0
 10    HWY_DIST           +$10,000        +$30,000        $0
```

Units: dollars per 1 standard deviation of each (Z-score standardized) feature.

### Prediction Matrix

- **W shape:** (11, 3)
- **P = XW shape:** (13932, 3)

**Interpretation of P[4, 1] = $720,832.92:**
This is the predicted sale price of house #5 (0-indexed: house index 4) under Scenario 2 ("Location is Everything"). The location-dominant model predicts this house would sell for $720,832.92.

**Advantage of XW over three separate products:**
Computing P = XW produces all three prediction vectors simultaneously in a single matrix operation. This is more computationally efficient (one BLAS call vs. three) and more mathematically elegant — it makes explicit that the three scenarios share the same data matrix X and differ only in their weight columns.

---

## 5. Phase 3: Manual Gate — Coefficient Translation

**Feature audited:** TOT_LVG_AREA, Scenario 1

**Scaling applied:** Z-score standardization → z = (x − μ) / σ, so α = 1/σ, β = −μ/σ

**Data statistics:**
- μ (mean) = 2058.0446 sq ft
- σ (std dev) = 813.5385 sq ft
- α = 1/σ = 0.001229
- β = −μ/σ = −2.529744

**Coefficient in standardized space:** w_z = $80,000 (per 1 SD of living area)

**Translated to original units:**
- w̃_x = α · w_z = 0.001229 × $80,000 = **$98.34 per square foot**
- Δw₀ = β · w_z = −2.529744 × $80,000 = **−$202,379.55** (shift to intercept)

**Interpretation:** In Scenario 1, each additional square foot of living area adds approximately **$98.34** to the predicted sale price.

**Verification:** This aligns with reasonable Miami construction/land cost estimates (~$100/sqft range for baseline linear approximation).

---

## 6. Phase 4: Valuation Audit (Partitioned Matrices)

### Block Decomposition of X

X = [ **1** | X_struc | X_loc ]

| Block | Columns | Content |
|---|---|---|
| **1** | 0 | Intercept column (all ones) |
| X_struc | 1–5 | TOT_LVG_AREA, LND_SQFOOT, structure_quality, age, SPEC_FEAT_VAL |
| X_loc | 6–10 | OCEAN_DIST, WATER_DIST, CNTR_DIST, SUBCNTR_DI, HWY_DIST |

### Block Formula Explanation

By the rules of partitioned matrix multiplication, since X = [**1** | X_struc | X_loc] and w = [w₀; w_struc; w_loc]:

**ŷ = w₀·1 + X_struc·w_struc + X_loc·w_loc**

Each block multiplies only its corresponding sub-vector of w. The three terms are additive and interpretable:
- **Base (w₀·1):** A flat $350,000 baseline applied identically to every house
- **Structure Value (X_struc·w_struc):** The premium or discount driven purely by physical property attributes
- **Location Value (X_loc·w_loc):** The premium or discount driven purely by geographic position

### Breakdown Table — First 5 Houses (Scenario 1, Baseline)

| House Index | Actual ($) | Base ($) | Struc Val ($) | Loc Val ($) | Predicted ($) | Error ($) |
|---|---|---|---|---|---|---|
| 0 | 440,000 | 350,000 | −55,322 | 114,377 | 409,055 | +30,945 |
| 1 | 349,000 | 350,000 | −55,277 | 124,464 | 419,187 | −70,187 |
| 2 | 800,000 | 350,000 | +108,049 | 124,877 | 582,926 | +217,074 |
| 3 | 988,000 | 350,000 | +12,785 | 126,925 | 489,710 | +498,290 |
| 4 | 755,000 | 350,000 | +11,662 | 123,312 | 484,974 | +270,026 |

**Partition verification:** partition sum == X @ w1: **PASS ✓**

### Interpretation

The decomposition reveals that for these 5 houses, **location is the dominant positive contributor** — all five have positive location values driven by proximity to ocean/water. Structure value varies widely: houses 0 and 1 have negative structure values (older or smaller below-average properties), while house 2's structure adds over $100k. This decomposition makes it possible to diagnose *why* a model under- or over-values a specific property.

---

## 7. Phase 5: Least Squares Benchmark

### Part A — Underdetermined Case

X_small ∈ ℝ^(5×11): 5 equations, 11 unknowns.

**Answer:** There are **infinitely many solutions**. With fewer equations than unknowns, the system is underdetermined — there is an entire affine subspace of weight vectors that satisfy X_small·w = y_small exactly. We cannot identify a unique solution without additional constraints.

### Part B — Optimal Weights (Full Dataset)

Least-squares solution computed via `np.linalg.lstsq(X, y, rcond=None)`.

**Full LS weight vector:**

| Feature | LS Weight | Scenario 1 (Manual) |
|---|---|---|
| Intercept | $399,941.93 | $350,000.00 |
| TOT_LVG_AREA | $161,561.06 | $80,000.00 |
| LND_SQFOOT | $16,373.42 | $25,000.00 |
| structure_quality | $69,029.63 | $60,000.00 |
| age | −$40,151.34 | −$20,000.00 |
| SPEC_FEAT_VAL | $41,333.11 | $30,000.00 |
| OCEAN_DIST | −$79,097.90 | −$55,000.00 |
| WATER_DIST | +$2,800.91 | −$20,000.00 |
| CNTR_DIST | −$96,269.40 | −$25,000.00 |
| SUBCNTR_DI | +$9,640.99 | −$15,000.00 |
| HWY_DIST | +$24,330.68 | +$10,000.00 |

**Comparison — TOT_LVG_AREA:** The LS weight is **$161,561** vs. the manual choice of **$80,000**. The data-driven weight is approximately **2× higher**, suggesting the manual choice significantly undervalued living area. Notably, WATER_DIST and SUBCNTR_DI flip sign — the data suggests these have counterintuitive effects once other features are controlled for.

### Part C — Intercept Experiment

| Model | MAE |
|---|---|
| Least Squares (with intercept) | $115,510 |
| Least Squares (no intercept) | $400,844 |
| **Difference** | **+$285,334** |

**Explanation:** Without the intercept, the model is forced to pass through the origin — predicting $0 for a house with all-zero standardized features. In real estate, the "all-average house" still sells for ~$400,000 (the market baseline). Removing the intercept forces the model to compensate by distorting all feature coefficients to approximate that baseline, resulting in severe misfits across the dataset. The $285,334 increase in MAE demonstrates that the intercept is not cosmetic — it is load-bearing.

---

## 8. Evaluation: MAE Scoreboard

| Model | MAE |
|---|---|
| Scenario 1 — Baseline (Rational Market) | **$112,587** |
| Scenario 2 — Location is Everything | $250,389 |
| Scenario 3 — The Luxury Premium | $181,319 |
| Least Squares (Optimal) | $115,510 |

> Bar chart saved to `mae_scoreboard.png`

**Key observations:**
- Scenario 1 (Baseline) performs nearly as well as Least Squares ($112,587 vs. $115,510) — a difference of only ~$3k MAE. This suggests the manual weight choices were well-calibrated.
- Scenario 2 (Location is Everything) has the worst MAE ($250,389), showing that ignoring structural features causes severe underfitting — many high-value homes are undervalued because build quality is ignored.
- Scenario 3 (Luxury Premium) falls in between, reasonably capturing ocean-adjacent luxury homes but overpricing average-quality inland properties.
- Least Squares is optimal by definition — it minimizes squared error and therefore also tends to minimize MAE.

---

## 9. Final Reflection

The essential Chapter 2 ideas in this project were **matrix-vector multiplication** (ŷ = Xw), **matrix-matrix multiplication** (P = XW), **dimension compatibility**, and **partitioned/block matrices**. Every phase reduced to a question of what shape a matrix product would take and what each entry of that product meant in the housing context.

The product Xw computes one prediction vector under a single pricing policy. The matrix product XW extends this to three policies simultaneously — each column of W encodes a different market hypothesis, and the corresponding column of P = XW gives every house's predicted price under that scenario. This allows side-by-side comparison across all 13,932 houses without repeating the computation. The partitioned-matrix decomposition added interpretive power: by splitting X into [**1** | X_struc | X_loc], we could decompose any prediction into a base component, a structure-driven component, and a location-driven component — making it possible to say not just *what* a model predicts, but *why*. The least-squares comparison was the most instructive check: despite choosing weights by intuition alone, Scenario 1 came within $3,000 MAE of the data-optimal solution, validating the economic logic behind the choices. It also revealed where intuition failed — the data values TOT_LVG_AREA roughly twice as highly as the manual estimate, and flips the sign on WATER_DIST and SUBCNTR_DI once multicollinearity is resolved.
