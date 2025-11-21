"""Core Synthetic Control (SCM) implementation.

This module provides:
- compute_scm_weights: augmented SCM weight optimization
- build_synthetic_unit: log-rescaled reconstruction of synthetic outcome

The implementation is fully generic and anonymized.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def compute_scm_weights(
    panel_df,
    predictors_df,
    predictor_cols,
    treated_unit,
    unit_col="unit",
    time_col="time",
    outcome_col="log_outcome",
    treatment_time=None,
    donor_units=None,
    lambda_cov=1.0,
    gamma=0.1,
    tau=0.05,
    cov_weights=None,
    lb=0.0,
    ub=1.0,
):
    """Compute donor weights for Synthetic Control.

    Minimizes an augmented loss:

        L = outcome_mse
            + lambda_cov * covariate_mse
            + gamma * L2_reg (toward uniform)
            + tau * entropy_penalty

    Parameters
    ----------
    panel_df : DataFrame
        Long-format panel data with columns [unit_col, time_col, outcome_col].
    predictors_df : DataFrame
        DataFrame with predictor values per unit per time.
    predictor_cols : list[str]
        Names of predictor features to balance.
    treated_unit : str
        The unit receiving treatment.
    unit_col : str, default "unit"
        Name of the column for units.
    time_col : str, default "time"
        Name of the column for time.
    outcome_col : str, default "log_outcome"
        Name of the outcome column (log-transformed recommended).
    treatment_time : str or datetime-like
        Time period at which treatment begins (pre < treatment_time).
    donor_units : list[str] or None
        Units eligible to be donors. If None, all other units are used.
    lambda_cov, gamma, tau : float
        Regularization hyperparameters.
    cov_weights : list or None
        Feature-specific weights for covariate balancing.
    lb, ub : float
        Bounds for donor weights.

    Returns
    -------
    donors : list[str]
        List of donor unit names.
    weights : np.ndarray
        Optimized weights (sum to 1).
    used_pre_periods : Index
        Time periods used in pre-treatment fitting.
    diagnostics : dict
        Contains pre-period error metrics and covariate balance info.
    """

    df = panel_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    if treatment_time is None:
        raise ValueError("treatment_time must be provided")

    treat_t = pd.to_datetime(treatment_time)

    # Pre-treatment subset
    pre_df = df[df[time_col] < treat_t]
    Y = pre_df.pivot(index=time_col, columns=unit_col, values=outcome_col).sort_index()

    if treated_unit not in Y.columns:
        raise ValueError("Treated unit not found in panel_df")

    all_units = [u for u in Y.columns if u != treated_unit]
    if donor_units is None:
        donors = all_units
    else:
        donors = [u for u in donor_units if u in all_units]

    if len(donors) < 1:
        raise ValueError("No valid donor units available")

    # Drop any rows with missing values across treated + donors
    Y_sub = Y[[treated_unit] + donors].dropna(how="any")
    used_pre_periods = Y_sub.index

    if len(used_pre_periods) < 3:
        raise ValueError("Not enough pre-treatment periods with complete data")

    y_t = Y_sub[treated_unit].to_numpy()
    X_t = Y_sub[donors].to_numpy()
    k = len(donors)

    # Predictor balancing
    P = predictors_df.copy()
    P[time_col] = pd.to_datetime(P[time_col])
    P_pre = P[P[time_col] < treat_t]

    P_pre[predictor_cols] = P_pre[predictor_cols].fillna(0.0)

    mu = P_pre[predictor_cols].mean()
    sd = P_pre[predictor_cols].std(ddof=0).replace(0, 1.0)

    Pz = (P_pre[predictor_cols] - mu) / sd
    P_mu = Pz.groupby(unit_col).mean()

    if treated_unit not in P_mu.index:
        raise ValueError("Predictors missing pre-period for treated unit")

    donors = [d for d in donors if d in P_mu.index]
    if len(donors) < 1:
        raise ValueError("No donors left after aligning predictors")

    X_t = Y_sub[donors].to_numpy()

    p_treated = P_mu.loc[treated_unit, predictor_cols].to_numpy()
    P_don_mat = P_mu.loc[donors, predictor_cols].to_numpy().T

    # Covariate weights
    if cov_weights is None:
        cov_w = np.ones(len(predictor_cols))
    else:
        cov_w = np.asarray(cov_weights, dtype=float)

    # Initial weights and constraints
    w0 = np.ones(len(donors)) / len(donors)
    bounds = [(lb, ub)] * len(donors)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def loss(w):
        # Outcome fit
        mse_y = np.mean((y_t - X_t @ w) ** 2)
        # Covariate balance
        cov_diff = p_treated - (P_don_mat @ w)
        mse_cov = np.mean(cov_w * (cov_diff ** 2))
        # L2 reg toward uniform
        reg_l2 = np.sum((w - 1.0 / len(donors)) ** 2)
        # Entropy reg
        reg_entropy = np.sum(w * np.log(w + 1e-12))
        return mse_y + lambda_cov * mse_cov + gamma * reg_l2 + tau * reg_entropy

    res = minimize(loss, w0, bounds=bounds, constraints=constraints)

    if not res.success:
        print("WARNING: optimizer did not fully converge:", res.message)

    w = res.x

    diagnostics = {
        "pre_outcome_mse": float(np.mean((y_t - X_t @ w) ** 2)),
        "pre_cov_mse": float(
            np.mean(((p_treated - (P_don_mat @ w)) ** 2) * cov_w)
        ),
        "predictor_mu_treated": p_treated,
        "predictor_mu_synth": (P_don_mat @ w),
        "donors": donors,
    }

    return donors, w, used_pre_periods, diagnostics


def build_synthetic_unit(
    panel_df,
    treated_unit,
    donors,
    weights,
    treatment_time,
    unit_col="unit",
    time_col="time",
    outcome_col="log_outcome",
):
    """Build synthetic outcome for a treated unit with log-rescaled reconstruction.

    Steps:
    1. Construct synthetic outcome in log space using donor units and weights.
    2. In the pre-treatment period, estimate a shift term c_hat so that:
           treated_log ≈ synthetic_log + c_hat
    3. Apply c_hat to synthetic_log and exponentiate both treated and synthetic
       to obtain level-space outcomes.

    Returns
    -------
    merged : DataFrame
        Columns: [time, synthetic_log, treated_log, synthetic_rescaled,
                  treated, synthetic]
    c_hat : float
        Estimated average log-difference in the pre period.
    """

    df = panel_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array")

    if len(weights) != len(donors):
        raise ValueError("weights and donors length mismatch")

    all_times = sorted(df[time_col].unique())
    syn_records = []

    for t in all_times:
        donor_vals = []
        for d in donors:
            tmp = df[(df[unit_col] == d) & (df[time_col] == t)]
            if tmp.empty:
                donor_vals.append(np.nan)
            else:
                donor_vals.append(float(tmp[outcome_col].iloc[0]))

        donor_vals = np.array(donor_vals, float)
        valid = np.isfinite(donor_vals)

        if valid.any():
            w_sub = weights[valid]
            if w_sub.sum() > 0:
                w_sub = w_sub / w_sub.sum()
                syn_val = float(np.sum(donor_vals[valid] * w_sub))
            else:
                syn_val = np.nan
        else:
            syn_val = np.nan

        syn_records.append({time_col: t, "synthetic_log": syn_val})

    syn_df = pd.DataFrame(syn_records).sort_values(time_col)

    treated = (
        df[df[unit_col] == treated_unit][[time_col, outcome_col]]
        .rename(columns={time_col: "time", outcome_col: "treated_log"})
    )

    merged = syn_df.rename(columns={time_col: "time"}).merge(
        treated, on="time", how="left"
    )

    treat_t = pd.to_datetime(treatment_time)
    pre_mask = merged["time"] < treat_t

    valid_pre = (
        pre_mask
        & merged["synthetic_log"].notna()
        & merged["treated_log"].notna()
    )

    if valid_pre.any():
        c_hat = (
            merged.loc[valid_pre, "treated_log"]
            - merged.loc[valid_pre, "synthetic_log"]
        ).mean()
    else:
        c_hat = 0.0

    merged["synthetic_rescaled"] = merged["synthetic_log"] + c_hat

    # Convert back to level space
    merged["treated"] = np.expm1(merged["treated_log"])
    merged["synthetic"] = np.expm1(merged["synthetic_rescaled"])

    return merged.sort_values("time"), float(c_hat)
