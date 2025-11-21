# Synthetic Control Toolkit
*A clean, modular, production-ready implementation of an augmented Synthetic Control Method — with covariate balancing, entropy regularization, log-rescaled reconstruction, and placebo testing.*

Synthetic Control (SCM) is a powerful causal inference technique used to estimate counterfactual outcomes when controlled experiments are not feasible.  
This toolkit provides a **generic, reusable, and fully anonymized** implementation suitable for:

- product experimentation  
- market-level intervention analysis  
- policy evaluation  
- attribution modeling  
- synthetic baselines for global launches  
- uplift estimation for region-specific programs  

This repository contains **no company-specific or proprietary context** and is designed as an open, academic-style implementation.

---

## 🚀 Features

### ✓ Augmented SCM loss function
Includes:
- outcome MSE  
- covariate balancing  
- L2 regularization toward uniform weights  
- entropy regularization (encourages diverse donor weights)

### ✓ Log-rescaled reconstruction
Handles large scale differences between units by:
- fitting SCM in log space  
- estimating a scaling term  
- reconstructing synthetic outcomes in level space

### ✓ Full placebo testing module
Evaluate robustness by treating every unit as if it were "pseudo-treated" and comparing post/pre RMSE ratios.

### ✓ Clean, modular architecture
Functions:
- `compute_scm_weights()`  
- `build_synthetic_unit()`  
- `scm_placebo()`  

Designed to be plugged directly into:
- ML pipelines  
- causal inference workflows  
- product analytics workflows  
- experimentation systems

### ✓ 100% generic & safe
No references to:
- real products  
- real markets  
- internal schema  
- dates or events  

---

## 📦 Installation

Clone the repository:

```bash
git clone [https://github.com/<your-username>/synthetic-control-toolkit.git](https://github.com/ansel-lin-global/synthetic-control-toolkit.git)
cd synthetic-control-toolkit
```

(Optionally) install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📘 Quick Start Example

Below is a minimal working example using your own or simulated data.

```python
import pandas as pd
from src.scm_core import compute_scm_weights, build_synthetic_unit

# Example panel data (structure only)
# columns: [unit, time, log_outcome]
panel = pd.read_csv("examples/simulated_panel_data.csv")

# Example predictors data
# columns: [unit, time, feature_a, feature_b, feature_c]
predictors = pd.read_csv("examples/simulated_predictors.csv")

predictor_cols = ["feature_a", "feature_b", "feature_c"]

donors, w, used_periods, diag = compute_scm_weights(
    panel_df=panel,
    predictors_df=predictors,
    predictor_cols=predictor_cols,
    treated_unit="Unit_0",
    treatment_time="2024-01-01",
    outcome_col="log_outcome",
    lambda_cov=2.0,
    gamma=0.1,
    tau=0.05
)

synthetic_df, c_hat = build_synthetic_unit(
    panel_df=panel,
    treated_unit="Unit_0",
    donors=donors,
    weights=w,
    treatment_time="2024-01-01"
)

print(synthetic_df.head())
```

---

## 🔍 Placebo Test Example

Produce post/pre RMSE ratios for all units:

```python
from src.scm_placebo import scm_placebo

units = panel["unit"].unique().tolist()

placebo_df = scm_placebo(
    panel_df=panel,
    predictors_df=predictors,
    units=units,
    predictor_cols=predictor_cols,
    treatment_time="2024-01-01"
)

print(placebo_df)
```

You can visualize placebo distributions to check robustness.

---

## 📊 Example Outputs

### Synthetic vs Actual (log-rescaled)

```text
time      treated      synthetic
---------------------------------
t0        134          128
t1        210          150
t2        320          172
...
```

### Placebo RMSE ratio illustration

```text
unit        ratio
---------------------
Unit_0      87.6    (treated)
Unit_1      40.0
Unit_2      29.1
Unit_3      14.5
...
```

---

## 🧠 Methodology Overview

### 1. Synthetic Control
Construct a counterfactual outcome using a weighted combination of donor units.

### 2. Augmented loss function
SCM is extended with:
- trend matching  
- covariate matching  
- entropy regularization  
- L2 regularization  

### 3. Log-rescale
Stabilizes cross-unit heterogeneity:
- training in log space  
- reconstructing level outcomes with a scaling term  

### 4. Placebo analysis
Standard SCM robustness test:  
large outlier ratios indicate true causal treatment effects.

---

## 📁 Repository Structure

```text
synthetic-control-toolkit/
├── src/
│   ├── scm_core.py          # SCM weights + log-rescale
│   ├── scm_placebo.py       # Placebo testing module
│   └── __init__.py
├── examples/
│   ├── demo_scm.ipynb       # Interactive example (optional)
│   ├── simulated_panel_data.csv
│   └── simulated_predictors.csv
├── README.md
└── LICENSE
```

---

## 🌍 Why this toolkit?

Real-world analytics often require causal inference when:

- A/B testing is impossible  
- regions differ massively in size  
- global launches create structural breaks  
- interventions occur only in a subset of markets  
- historical baselines don’t exist  

This toolkit offers a **clean, practical implementation** that mirrors what’s used in large-scale product organizations.

---

## ✨ Author

**Ansel Lin**  
Product-minded Data Scientist | Causal Inference | ML Systems

