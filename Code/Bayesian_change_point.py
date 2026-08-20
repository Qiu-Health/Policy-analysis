# -*- coding: utf-8 -*-
"""Combined Bayesian change-point analysis for PFAS temporal trends.

This single script contains both:
1. the primary Bayesian analysis used for the manuscript, using the
   0.5- to 1.5-fold annual-change slope prior; and
2. slope-prior sensitivity analyses using alternative 0.8- to 1.2-fold
   and 0.1- to 1.9-fold annual-change priors.

All scenarios use the same:
- 9 PFAS analytes
- hierarchical monthly median aggregation
- log10 transformation
- rolling single-pass Grubbs screening (±1 year, alpha=0.05)
- 0-change-point and 1-change-point Bayesian models
- PyMC MCMC settings
- ArviZ diagnostics and LOO-CV
- ELPD/SE model-selection rule
- result-export workflow

The primary scenario is run once and is reused in the combined sensitivity
summary, so it is not needlessly repeated.
"""

import os

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy import stats
from scipy.stats import gaussian_kde


# =============================================================================
# Configuration
# =============================================================================

INPUT_CSV = r"D:\LinJiayi\data\41PFAS_limit_ge1.csv"
BASE_OUTPUT_DIR = r"D:\LinJiayi\data\Bayesion\limitGE1_Grubbs3_month_pollute-1"

# Keep this list synchronized with the analytes used for the final manuscript run.
SUBSTANCES = [
    "PFDoDA",
    "PFTeDA",
    "PFOcDA",
    "PFUnDA",
    "8:2 FTSA",
    "PFDA",
    "PFNA",
    "PFHxS",
    "PFOA",
]
# Rolling Grubbs screening
GRUBBS_WINDOW_YEARS = 1
GRUBBS_ALPHA = 0.05
GRUBBS_MIN_N = 3

# Bayesian sampling
RANDOM_SEED = 123
DRAWS = 2000
TUNE = 1000
N_CHAINS = 4
TARGET_ACCEPT = 0.95
SIGMA_PRIOR = 1.0
MIN_MODEL_POINTS = 8

# Diagnostics / model selection
RHAT_THRESHOLD = 1.15
LOO_Z_THRESHOLD = 1.96

# Slope-prior scenarios
SLOPE_PRIOR_OPTIONS = {
    "slope_prior_20": (np.log10(0.8), np.log10(1.2)),
    "slope_prior_50": (np.log10(0.5), np.log10(1.5)),
    "slope_prior_90": (np.log10(0.1), np.log10(1.9)),
}

# Primary analysis used in the manuscript
PRIMARY_SLOPE_PRIOR_SCENARIO = "slope_prior_50"


# =============================================================================
# Utilities
# =============================================================================

def hdi_bounds(samples, prob=0.95):
    """Return lower and upper bounds of an HDI."""
    low, high = az.hdi(np.asarray(samples), hdi_prob=prob)
    return float(low), float(high)


def hdi_bounds_axis0(samples_2d, prob=0.95):
    """Compute an HDI for each prediction point."""
    samples_2d = np.asarray(samples_2d)
    bounds = np.array(
        [hdi_bounds(samples_2d[:, j], prob) for j in range(samples_2d.shape[1])]
    )
    return bounds[:, 0], bounds[:, 1]


def posterior_mode(samples):
    """Estimate posterior mode using a Gaussian KDE."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]

    if len(samples) < 2:
        return np.nan
    if np.std(samples) == 0:
        return float(samples[0])

    kde = gaussian_kde(samples)
    grid = np.linspace(samples.min(), samples.max(), 1000)
    return float(grid[np.argmax(kde(grid))])


def annual_percent_change(slope_samples):
    """Convert log10 slope per year to annual percentage change."""
    return (10 ** np.asarray(slope_samples) - 1) * 100


# =============================================================================
# Rolling Grubbs outlier screening
# =============================================================================

def grubbs_critical_value(n, alpha=0.05):
    t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    return ((n - 1) / np.sqrt(n)) * np.sqrt(
        t_crit**2 / (n - 2 + t_crit**2)
    )


def grubbs_one_iteration(values, alpha=0.05):
    """Perform one non-iterative Grubbs test; return local outlier index."""
    x = np.asarray(values, dtype=float)
    valid = ~np.isnan(x)
    x_valid = x[valid]

    n = len(x_valid)
    if n < 3:
        return None

    std = np.std(x_valid, ddof=1)
    if std == 0:
        return None

    deviations = np.abs(x_valid - np.mean(x_valid))
    max_idx_valid = int(np.argmax(deviations))
    g_stat = deviations[max_idx_valid] / std

    if g_stat > grubbs_critical_value(n, alpha):
        return int(np.where(valid)[0][max_idx_valid])

    return None


def rolling_month_single_pass_grubbs(group):
    """Flag local outliers using a ±1-year rolling window."""
    group = group.sort_values("date").copy().reset_index(drop=True)
    group["grubbs_outlier"] = False
    candidates = set()

    for i in range(len(group)):
        center_date = group.loc[i, "date"]
        start_date = center_date - pd.DateOffset(years=GRUBBS_WINDOW_YEARS)
        end_date = center_date + pd.DateOffset(years=GRUBBS_WINDOW_YEARS)
        in_window = group["date"].between(start_date, end_date)

        if in_window.sum() < GRUBBS_MIN_N:
            continue

        local_idx = grubbs_one_iteration(
            group.loc[in_window, "log_value_for_grubbs"].to_numpy(),
            alpha=GRUBBS_ALPHA,
        )

        if local_idx is not None:
            candidates.add(group.index[in_window][local_idx])

    if candidates:
        group.loc[list(candidates), "grubbs_outlier"] = True

    return group


# =============================================================================
# Data preparation
# =============================================================================

def prepare_monthly_data(input_csv, substances):
    """Read, aggregate and screen monthly PFAS data."""
    df = pd.read_csv(input_csv)
    df = df[df["Substance"].isin(substances)].copy()

    df["value [ng/L]"] = pd.to_numeric(df["value [ng/L]"], errors="coerce")
    df["limit [ng/L]"] = pd.to_numeric(df["limit [ng/L]"], errors="coerce")
    df["value [ng/L]"] = df["value [ng/L]"].fillna(
        df["limit [ng/L]"] / np.sqrt(2)
    )

    df["lon_grid"] = np.floor(df["lon"]).astype(int)
    df["lat_grid"] = np.floor(df["lat"]).astype(int)

    # Hierarchical medians:
    # sampling point -> source/grid -> grid -> global monthly median
    median_raw = (
        df.groupby(
            [
                "year", "month", "source", "lon", "lat",
                "lon_grid", "lat_grid", "Substance"
            ]
        )["value [ng/L]"]
        .median()
        .reset_index()
    )

    median_cell = (
        median_raw.groupby(
            ["year", "month", "source", "lon_grid", "lat_grid", "Substance"]
        )["value [ng/L]"]
        .median()
        .reset_index()
    )

    median_grid = (
        median_cell.groupby(
            ["year", "month", "lon_grid", "lat_grid", "Substance"]
        )["value [ng/L]"]
        .median()
        .reset_index()
    )

    monthly = (
        median_grid.groupby(["year", "month", "Substance"])["value [ng/L]"]
        .median()
        .reset_index()
    )

    monthly["date"] = pd.to_datetime(
        monthly[["year", "month"]].assign(day=1)
    )
    monthly = monthly.sort_values(["Substance", "date"]).reset_index(drop=True)
    monthly = monthly[monthly["value [ng/L]"] > 0].copy()
    monthly["log_value_for_grubbs"] = np.log10(monthly["value [ng/L]"])

    screened = pd.concat(
        [
            rolling_month_single_pass_grubbs(group)
            for _, group in monthly.groupby("Substance")
        ],
        ignore_index=True,
    )

    outlier_report = (
        screened.groupby("Substance")
        .agg(
            n_total=("grubbs_outlier", "size"),
            n_removed=("grubbs_outlier", "sum"),
        )
        .reset_index()
    )
    outlier_report["removed_percent"] = (
        outlier_report["n_removed"] / outlier_report["n_total"] * 100
    )

    clean = screened[~screened["grubbs_outlier"]].copy().reset_index(drop=True)
    clean["log_value"] = np.log10(clean["value [ng/L]"])
    clean["time"] = (
        clean["year"].astype(float) + (clean["month"] - 1) / 12
    )

    return screened, clean, outlier_report


# =============================================================================
# Bayesian models
# =============================================================================

def fit_no_cp_model(d, slope_lower, slope_upper):
    """Fit the no-change-point linear Bayesian model."""
    d = d.sort_values("time").copy()
    x = d["time"].to_numpy()
    y = d["log_value"].to_numpy()
    x_center = x - x.mean()

    with pm.Model():
        intercept = pm.Uniform(
            "intercept",
            lower=y.min() - 5,
            upper=y.max() + 5,
        )
        slope = pm.Uniform(
            "slope",
            lower=slope_lower,
            upper=slope_upper,
        )
        sigma = pm.HalfNormal("sigma", sigma=SIGMA_PRIOR)

        mu = intercept + slope * x_center
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=N_CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

    return trace


def fit_one_cp_model(d, slope_lower, slope_upper):
    """Fit the one-change-point segmented Bayesian model."""
    d = d.sort_values("time").copy()
    x = d["time"].to_numpy()
    y = d["log_value"].to_numpy()
    x_center = x - x.mean()

    with pm.Model():
        intercept = pm.Uniform(
            "intercept",
            lower=y.min() - 5,
            upper=y.max() + 5,
        )
        slope_before = pm.Uniform(
            "slope_before",
            lower=slope_lower,
            upper=slope_upper,
        )
        slope_after = pm.Uniform(
            "slope_after",
            lower=slope_lower,
            upper=slope_upper,
        )
        cp = pm.Uniform(
            "cp",
            lower=x_center.min(),
            upper=x_center.max(),
        )
        sigma = pm.HalfNormal("sigma", sigma=SIGMA_PRIOR)

        mu = (
            intercept
            + slope_before * x_center
            + (slope_after - slope_before)
            * pt.maximum(0, x_center - cp)
        )

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=N_CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

    return trace


def trace_diagnostics(trace, var_names):
    """Return key MCMC diagnostics."""
    summary = az.summary(trace, var_names=var_names)

    return {
        "summary": summary,
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "n_divergences": int(
            trace.sample_stats["diverging"].values.sum()
        ),
    }


def loo_difference(loo_no_cp, loo_one_cp):
    """Return ELPD(one CP - no CP), paired SE and z score."""
    diff = float(loo_one_cp.elpd_loo - loo_no_cp.elpd_loo)

    pointwise_diff = (
        loo_one_cp.loo_i.values - loo_no_cp.loo_i.values
    )
    se = float(
        np.sqrt(
            len(pointwise_diff)
            * np.var(pointwise_diff, ddof=1)
        )
    )
    z = diff / se if se > 0 else np.nan

    return diff, se, z


def select_model(loo_diff, loo_z):
    """Select 1-CP only if ELPD improves and z >= 1.96."""
    if (
        np.isfinite(loo_z)
        and loo_diff > 0
        and loo_z >= LOO_Z_THRESHOLD
    ):
        return "one_change_point"

    return "no_change_point"


def fit_all_substances(
    clean_data,
    slope_prior_scenario,
    slope_lower,
    slope_upper,
):
    """Fit both candidate models for each PFAS."""
    fits = {}
    loo_rows = []

    for substance, d in clean_data.groupby("Substance"):
        d = d.sort_values("time").copy()

        if len(d) < MIN_MODEL_POINTS:
            print(
                f"{substance}: fewer than "
                f"{MIN_MODEL_POINTS} time points; skipped"
            )
            continue

        print(f"Fitting {substance}...")

        trace_0 = fit_no_cp_model(
            d,
            slope_lower,
            slope_upper,
        )
        trace_1 = fit_one_cp_model(
            d,
            slope_lower,
            slope_upper,
        )

        diag_0 = trace_diagnostics(
            trace_0,
            ["intercept", "slope", "sigma"],
        )
        diag_1 = trace_diagnostics(
            trace_1,
            [
                "intercept", "slope_before",
                "slope_after", "cp", "sigma"
            ],
        )

        loo_0 = az.loo(trace_0, pointwise=True)
        loo_1 = az.loo(trace_1, pointwise=True)

        loo_diff, loo_se, loo_z = loo_difference(
            loo_0,
            loo_1,
        )
        selected_model = select_model(
            loo_diff,
            loo_z,
        )

        fits[substance] = {
            "data": d,
            "trace_0": trace_0,
            "trace_1": trace_1,
            "loo_0": loo_0,
            "loo_1": loo_1,
            "diag_0": diag_0,
            "diag_1": diag_1,
            "selected_model": selected_model,
        }

        loo_rows.append(
            {
                "Substance": substance,
                "slope_prior_scenario": slope_prior_scenario,
                "slope_prior_lower": slope_lower,
                "slope_prior_upper": slope_upper,
                "n": len(d),
                "loo_no_cp": loo_0.elpd_loo,
                "loo_one_cp": loo_1.elpd_loo,
                "loo_diff_one_minus_no": loo_diff,
                "loo_diff_se": loo_se,
                "loo_z_score": loo_z,
                "final_selected_model": selected_model,
                "max_rhat_no_cp": diag_0["max_rhat"],
                "max_rhat_one_cp": diag_1["max_rhat"],
                "rhat_ok_no_cp": (
                    diag_0["max_rhat"] < RHAT_THRESHOLD
                ),
                "rhat_ok_one_cp": (
                    diag_1["max_rhat"] < RHAT_THRESHOLD
                ),
                "min_ess_bulk_no_cp": diag_0["min_ess_bulk"],
                "min_ess_bulk_one_cp": diag_1["min_ess_bulk"],
                "n_divergences_no_cp": diag_0["n_divergences"],
                "n_divergences_one_cp": diag_1["n_divergences"],
                "loo_warning_no_cp": bool(loo_0.warning),
                "loo_warning_one_cp": bool(loo_1.warning),
            }
        )

    return fits, pd.DataFrame(loo_rows)


# =============================================================================
# Result summaries
# =============================================================================

def parameter_row(
    substance,
    slope_prior_scenario,
    slope_lower,
    slope_upper,
    model,
    param,
    samples,
    rhat,
    ess,
    divergences,
    loo_row,
):
    low, high = hdi_bounds(samples)

    return {
        "PFAS": substance,
        "slope_prior_scenario": slope_prior_scenario,
        "slope_prior_lower": slope_lower,
        "slope_prior_upper": slope_upper,
        "Model": model,
        "param": param,
        "mean": float(np.mean(samples)),
        "mode": posterior_mode(samples),
        "lower_95_HDI": low,
        "upper_95_HDI": high,
        "Rhat": rhat,
        "ESS_bulk": ess,
        "n_divergences": divergences,
        "loo": (
            loo_row["loo_no_cp"]
            if model == "0 change point"
            else loo_row["loo_one_cp"]
        ),
        "loo_diff_one_minus_no": loo_row[
            "loo_diff_one_minus_no"
        ],
        "loo_diff_se": loo_row["loo_diff_se"],
        "z_score": loo_row["loo_z_score"],
        "final_selected_model": loo_row[
            "final_selected_model"
        ],
    }


def make_parameter_summary(
    fits,
    loo_summary,
    slope_prior_scenario,
    slope_lower,
    slope_upper,
):
    rows = []

    for substance, obj in fits.items():
        d = obj["data"]
        loo_row = loo_summary.loc[
            loo_summary["Substance"] == substance
        ].iloc[0]

        # 0-change-point model
        trace_0 = obj["trace_0"]
        diag_0 = obj["diag_0"]["summary"]

        p0 = {
            "intercept": trace_0.posterior[
                "intercept"
            ].values.flatten(),
            "slope": trace_0.posterior[
                "slope"
            ].values.flatten(),
            "sigma": trace_0.posterior[
                "sigma"
            ].values.flatten(),
        }

        for name, samples in p0.items():
            rows.append(
                parameter_row(
                    substance,
                    slope_prior_scenario,
                    slope_lower,
                    slope_upper,
                    "0 change point",
                    name,
                    samples,
                    diag_0.loc[name, "r_hat"],
                    diag_0.loc[name, "ess_bulk"],
                    obj["diag_0"]["n_divergences"],
                    loo_row,
                )
            )

        annual = annual_percent_change(p0["slope"])
        rows.append(
            parameter_row(
                substance,
                slope_prior_scenario,
                slope_lower,
                slope_upper,
                "0 change point",
                "annual_percent_change",
                annual,
                np.nan,
                np.nan,
                obj["diag_0"]["n_divergences"],
                loo_row,
            )
        )

        # 1-change-point model
        trace_1 = obj["trace_1"]
        diag_1 = obj["diag_1"]["summary"]

        cp_centered = trace_1.posterior[
            "cp"
        ].values.flatten()
        cp_year = cp_centered + d["time"].mean()

        slope_before = trace_1.posterior[
            "slope_before"
        ].values.flatten()
        slope_after = trace_1.posterior[
            "slope_after"
        ].values.flatten()
        slope_change = slope_after - slope_before

        p1 = {
            "intercept": trace_1.posterior[
                "intercept"
            ].values.flatten(),
            "cp_year": cp_year,
            "slope_before": slope_before,
            "slope_after": slope_after,
            "slope_change": slope_change,
            "sigma": trace_1.posterior[
                "sigma"
            ].values.flatten(),
        }

        for name, samples in p1.items():
            if name == "cp_year":
                rhat = diag_1.loc["cp", "r_hat"]
                ess = diag_1.loc["cp", "ess_bulk"]
            elif name in diag_1.index:
                rhat = diag_1.loc[name, "r_hat"]
                ess = diag_1.loc[name, "ess_bulk"]
            else:
                rhat = np.nan
                ess = np.nan

            rows.append(
                parameter_row(
                    substance,
                    slope_prior_scenario,
                    slope_lower,
                    slope_upper,
                    "1 change point",
                    name,
                    samples,
                    rhat,
                    ess,
                    obj["diag_1"]["n_divergences"],
                    loo_row,
                )
            )

        for name, samples in {
            "annual_percent_before": annual_percent_change(
                slope_before
            ),
            "annual_percent_after": annual_percent_change(
                slope_after
            ),
        }.items():
            rows.append(
                parameter_row(
                    substance,
                    slope_prior_scenario,
                    slope_lower,
                    slope_upper,
                    "1 change point",
                    name,
                    samples,
                    np.nan,
                    np.nan,
                    obj["diag_1"]["n_divergences"],
                    loo_row,
                )
            )

    return pd.DataFrame(rows)


# =============================================================================
# Origin-ready prediction data
# =============================================================================

def make_fit_line_data(substance, obj, n_grid=200):
    d = obj["data"]
    x = d["time"].to_numpy()
    x_mean = x.mean()

    x_grid = np.linspace(
        x.min(),
        x.max(),
        n_grid,
    )
    xc = x_grid - x_mean

    # 0 change point
    t0 = obj["trace_0"].posterior
    intercept = t0["intercept"].values.flatten()
    slope = t0["slope"].values.flatten()

    pred0 = (
        intercept[:, None]
        + slope[:, None] * xc[None, :]
    )
    low0, high0 = hdi_bounds_axis0(pred0)

    df0 = pd.DataFrame(
        {
            "Substance": substance,
            "model": "no_change_point",
            "time": x_grid,
            "log_pred_mean": pred0.mean(axis=0),
            "log_pred_low": low0,
            "log_pred_high": high0,
        }
    )

    # 1 change point
    t1 = obj["trace_1"].posterior
    intercept = t1["intercept"].values.flatten()
    slope_before = t1[
        "slope_before"
    ].values.flatten()
    slope_after = t1[
        "slope_after"
    ].values.flatten()
    cp = t1["cp"].values.flatten()

    pred1 = (
        intercept[:, None]
        + slope_before[:, None] * xc[None, :]
        + (slope_after - slope_before)[:, None]
        * np.maximum(
            0,
            xc[None, :] - cp[:, None],
        )
    )
    low1, high1 = hdi_bounds_axis0(pred1)

    df1 = pd.DataFrame(
        {
            "Substance": substance,
            "model": "one_change_point",
            "time": x_grid,
            "log_pred_mean": pred1.mean(axis=0),
            "log_pred_low": low1,
            "log_pred_high": high1,
        }
    )

    for frame in (df0, df1):
        frame["pred_mean"] = (
            10 ** frame["log_pred_mean"]
        )
        frame["pred_low"] = (
            10 ** frame["log_pred_low"]
        )
        frame["pred_high"] = (
            10 ** frame["log_pred_high"]
        )

    return df0, df1


# =============================================================================
# Export
# =============================================================================

def export_results(
    output_dir,
    screened,
    clean,
    outlier_report,
    fits,
    loo_summary,
    slope_prior_scenario,
    slope_lower,
    slope_upper,
):
    os.makedirs(output_dir, exist_ok=True)

    screened.to_csv(
        os.path.join(
            output_dir,
            "result_grubbs_all.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    screened[
        screened["grubbs_outlier"]
    ].to_csv(
        os.path.join(
            output_dir,
            "grubbs_outliers.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    outlier_report.to_csv(
        os.path.join(
            output_dir,
            "grubbs_outlier_report.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    clean.to_csv(
        os.path.join(
            output_dir,
            "median_year_month_clean_after_grubbs.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    loo_summary.to_csv(
        os.path.join(
            output_dir,
            "loo_model_comparison.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    parameter_summary = make_parameter_summary(
        fits,
        loo_summary,
        slope_prior_scenario,
        slope_lower,
        slope_upper,
    )

    parameter_summary.to_csv(
        os.path.join(
            output_dir,
            "model_parameter_summary.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    observed = clean[
        [
            "Substance",
            "year",
            "month",
            "time",
            "value [ng/L]",
            "log_value",
        ]
    ]

    observed.to_csv(
        os.path.join(
            output_dir,
            "origin_observed_points.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    no_cp_lines = []
    one_cp_lines = []

    for substance, obj in fits.items():
        df0, df1 = make_fit_line_data(
            substance,
            obj,
        )
        no_cp_lines.append(df0)
        one_cp_lines.append(df1)

    if no_cp_lines:
        pd.concat(
            no_cp_lines,
            ignore_index=True,
        ).to_csv(
            os.path.join(
                output_dir,
                "origin_model_0_no_cp_fit_lines.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )

        pd.concat(
            one_cp_lines,
            ignore_index=True,
        ).to_csv(
            os.path.join(
                output_dir,
                "origin_model_1_one_cp_fit_lines.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )

    return parameter_summary


# =============================================================================
# Reusable analysis runner
# =============================================================================

def run_analysis(
    slope_prior_scenario,
    input_csv=INPUT_CSV,
    substances=SUBSTANCES,
    base_output_dir=BASE_OUTPUT_DIR,
):
    """Run one complete Bayesian analysis under a specified slope-prior scenario."""
    if slope_prior_scenario not in SLOPE_PRIOR_OPTIONS:
        raise ValueError(
            f"Unknown slope-prior scenario: {slope_prior_scenario}"
        )

    slope_lower, slope_upper = SLOPE_PRIOR_OPTIONS[
        slope_prior_scenario
    ]
    output_dir = os.path.join(
        base_output_dir,
        slope_prior_scenario,
    )

    print("=" * 72)
    print(
        f"Slope-prior scenario: "
        f"{slope_prior_scenario}"
    )
    print(
        f"Slope prior: "
        f"[{slope_lower:.6f}, "
        f"{slope_upper:.6f}]"
    )
    print(
        f"Output directory: "
        f"{output_dir}"
    )

    screened, clean, outlier_report = (
        prepare_monthly_data(
            input_csv,
            substances,
        )
    )

    print("\nGrubbs screening summary:")
    print(outlier_report)

    fits, loo_summary = fit_all_substances(
        clean,
        slope_prior_scenario,
        slope_lower,
        slope_upper,
    )

    print("\nLOO model comparison:")
    print(loo_summary)

    parameter_summary = export_results(
        output_dir,
        screened,
        clean,
        outlier_report,
        fits,
        loo_summary,
        slope_prior_scenario,
        slope_lower,
        slope_upper,
    )

    print(
        f"\nExport complete: "
        f"{output_dir}"
    )

    return {
        "scenario": slope_prior_scenario,
        "output_dir": output_dir,
        "screened": screened,
        "clean": clean,
        "outlier_report": outlier_report,
        "fits": fits,
        "loo_summary": loo_summary,
        "parameter_summary": parameter_summary,
    }


# =============================================================================
# Combined primary + slope-prior sensitivity analysis
# =============================================================================

# Primary manuscript analysis
RUN_PRIMARY_ANALYSIS = True

# Alternative-prior sensitivity analyses
RUN_SLOPE_PRIOR_SENSITIVITY = True

# The primary 50% prior is not listed here because it is already run above.
SENSITIVITY_SCENARIOS = [
    "slope_prior_20",
    "slope_prior_90",
]

SENSITIVITY_SUMMARY_DIR = os.path.join(
    BASE_OUTPUT_DIR,
    "slope_prior_sensitivity_summary",
)


def export_sensitivity_summary(results):
    """Combine primary and alternative-prior results into comparison tables."""
    if not results:
        return

    os.makedirs(SENSITIVITY_SUMMARY_DIR, exist_ok=True)

    combined_loo = pd.concat(
        [result["loo_summary"].copy() for result in results],
        ignore_index=True,
    )

    combined_parameters = pd.concat(
        [result["parameter_summary"].copy() for result in results],
        ignore_index=True,
    )

    combined_loo.to_csv(
        os.path.join(
            SENSITIVITY_SUMMARY_DIR,
            "slope_prior_sensitivity_loo_comparison.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    combined_parameters.to_csv(
        os.path.join(
            SENSITIVITY_SUMMARY_DIR,
            "slope_prior_sensitivity_parameter_summary.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    # Compact table focused on whether model selection and diagnostics
    # change across slope-prior scenarios.
    model_selection = combined_loo[
        [
            "Substance",
            "slope_prior_scenario",
            "final_selected_model",
            "loo_no_cp",
            "loo_one_cp",
            "loo_diff_one_minus_no",
            "loo_diff_se",
            "loo_z_score",
            "max_rhat_no_cp",
            "max_rhat_one_cp",
            "min_ess_bulk_no_cp",
            "min_ess_bulk_one_cp",
            "n_divergences_no_cp",
            "n_divergences_one_cp",
            "loo_warning_no_cp",
            "loo_warning_one_cp",
        ]
    ].copy()

    model_selection.to_csv(
        os.path.join(
            SENSITIVITY_SUMMARY_DIR,
            "slope_prior_sensitivity_model_selection.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("Combined slope-prior sensitivity summaries exported to:")
    print(SENSITIVITY_SUMMARY_DIR)


def main():
    all_results = []

    # -------------------------------------------------------------------------
    # 1. Primary analysis: 0.5- to 1.5-fold annual-change slope prior
    # -------------------------------------------------------------------------
    if RUN_PRIMARY_ANALYSIS:
        print("\n" + "#" * 72)
        print("PRIMARY BAYESIAN ANALYSIS")
        print("#" * 72)

        primary_result = run_analysis(
            PRIMARY_SLOPE_PRIOR_SCENARIO
        )
        all_results.append(primary_result)

    # -------------------------------------------------------------------------
    # 2. Slope-prior sensitivity analyses
    # -------------------------------------------------------------------------
    if RUN_SLOPE_PRIOR_SENSITIVITY:
        print("\n" + "#" * 72)
        print("SLOPE-PRIOR SENSITIVITY ANALYSES")
        print("#" * 72)

        for scenario in SENSITIVITY_SCENARIOS:
            sensitivity_result = run_analysis(
                scenario
            )
            all_results.append(sensitivity_result)

        # Include the primary scenario in the comparison table if it was run.
        export_sensitivity_summary(all_results)

    print("\nAll requested analyses are complete.")


if __name__ == "__main__":
    main()
