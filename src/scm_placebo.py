"""Placebo testing utilities for Synthetic Control.

This module provides:
- scm_placebo: run SCM for each unit as if treated, compute post/pre RMSE ratio.
"""

import numpy as np
import pandas as pd

from .scm_core import compute_scm_weights, build_synthetic_unit


def _rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def scm_placebo(
    panel_df,
    predictors_df,
    units,
    predictor_cols,
    treatment_time,
    unit_col="unit",
    time_col="time",
    outcome_col="log_outcome",
    lambda_cov=1.0,
    gamma=0.1,
    tau=0.05,
    cov_weights=None,
):
    """Run SCM placebo tests across all units.

    For each unit in `units`, treat it as if it were the treated unit and
    estimate a synthetic control using the remaining units as donors.
    Then compute pre-/post-treatment RMSE and their ratio.

    Parameters
    ----------
    panel_df : DataFrame
        Panel data with [unit_col, time_col, outcome_col].
    predictors_df : DataFrame
        DataFrame with predictor values per unit per time.
    units : list[str]
        List of units to iterate over as placebos.
    predictor_cols : list[str]
        Predictor feature names.
    treatment_time : str or datetime-like
        Time period where treatment is assumed to start.
    unit_col : str, default "unit"
    time_col : str, default "time"
    outcome_col : str, default "log_outcome"
    lambda_cov, gamma, tau : float
        Hyperparameters for SCM.
    cov_weights : list or None
        Optional feature weights for covariate balancing.

    Returns
    -------
    DataFrame
        Columns include:
        - unit
        - donors
        - weights
        - pre_rmse
        - post_rmse
        - ratio (post / pre)
    """

    df = panel_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    results = []

    for treated in units:
        donor_pool = [u for u in units if u != treated]
        if len(donor_pool) < 1:
            continue

        try:
            donors, w, used_periods, diag = compute_scm_weights(
                panel_df=df,
                predictors_df=predictors_df,
                predictor_cols=predictor_cols,
                treated_unit=treated,
                unit_col=unit_col,
                time_col=time_col,
                outcome_col=outcome_col,
                treatment_time=treatment_time,
                donor_units=donor_pool,
                lambda_cov=lambda_cov,
                gamma=gamma,
                tau=tau,
                cov_weights=cov_weights,
            )
        except Exception:
            # Skip units that cannot be fit (e.g., insufficient data)
            continue

        syn_df, c_hat = build_synthetic_unit(
            panel_df=df,
            treated_unit=treated,
            donors=donors,
            weights=w,
            treatment_time=treatment_time,
            unit_col=unit_col,
            time_col=time_col,
            outcome_col=outcome_col,
        )

        t0 = pd.to_datetime(treatment_time)
        pre = syn_df[syn_df["time"] < t0]
        post = syn_df[syn_df["time"] >= t0]

        pre_rmse = _rmse(pre["treated"], pre["synthetic"])
        post_rmse = _rmse(post["treated"], post["synthetic"])

        ratio = np.nan
        if pre_rmse and pre_rmse > 1e-8:
            ratio = float(post_rmse / pre_rmse) if np.isfinite(post_rmse) else np.nan

        results.append(
            {
                "unit": treated,
                "donors": donors,
                "weights": w.tolist(),
                "pre_rmse": pre_rmse,
                "post_rmse": post_rmse,
                "ratio": ratio,
            }
        )

    return pd.DataFrame(results).sort_values("ratio", ascending=False)
