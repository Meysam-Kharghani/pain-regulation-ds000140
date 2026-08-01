#!/usr/bin/env python3
"""
Behavioral computational analysis pipeline for OpenNeuro ds000140.

Purpose
-------
This script assumes that the behavioral trial table has already been cleaned
and standardized, e.g. behavior_trials_master_clean.tsv. It intentionally does
not accept legacy residual or model-output columns as input. Instead, it recomputes
passive expected pain, residuals, trial-wise regulation success, asymmetry
indices, habituation/drift metrics, order effects, and model-comparison outputs
from the clean master table.

Main questions addressed
------------------------
1) Does explicit regulation change pain reports relative to passive experience?
2) Are up- and down-regulation equally successful relative to passive expectation?
3) Do pain reports or regulation success change within a run?
4) Do pain reports drift/habituate/sensitize across the 9-run session?
5) Does regulation order, up-first vs down-first, affect pain or regulation success?

Modeling conventions
--------------------
- "Up vs down separation" is reported as task separation rather than behavioral asymmetry.
- Direction-corrected behavioral asymmetry is tested on regulation-success measures:
      success_up   = observed - passive_expected
      success_down = passive_expected - observed
- Order effects are tested at the subject level, not as a rank-deficient
  trial-level condition × order model.
- GEE sensitivity uses Patsy design matrices with row-aligned cluster labels.
- The report separates validation, success, asymmetry, habituation, order,
  and computational model-comparison outputs.

Usage
-----
python run_behavior_models.py \
  --data behavior_trials_master_clean.tsv \
  --outdir behavior_computational

For a reduced validation run:
python run_behavior_models.py \
  --data behavior_trials_master_clean.tsv \
  --outdir behavior_computational_validation \
  --validation-run

Dependencies: numpy, pandas, scipy, statsmodels, matplotlib, patsy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.multitest import multipletests


# =============================================================================
# General utilities
# =============================================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def finite_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def zscore_by_group(x: pd.Series, group: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    def _z(v: pd.Series) -> pd.Series:
        s = v.std(ddof=0)
        if not np.isfinite(s) or s == 0:
            return v * 0.0
        return (v - v.mean()) / s
    return x.groupby(group, sort=False).transform(_z)


def mean_ci_boot(values: Sequence[float], n_boot: int = 5000, seed: int = 1) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = rng.choice(vals, size=vals.size, replace=True).mean()
    return tuple(np.percentile(boots, [2.5, 97.5]).astype(float))


def sign_flip_p(values: Sequence[float], n_perm: int = 20000, seed: int = 1) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    obs = abs(vals.mean())
    rng = np.random.default_rng(seed)
    sims = np.empty(n_perm)
    for b in range(n_perm):
        signs = rng.choice([-1, 1], size=vals.size, replace=True)
        sims[b] = abs((vals * signs).mean())
    return float((np.sum(sims >= obs) + 1) / (n_perm + 1))


def cohen_dz(values: Sequence[float]) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.nan
    sd = vals.std(ddof=1)
    return float(vals.mean() / sd) if sd > 0 else np.nan


def bh_fdr_table(df: pd.DataFrame, p_col: str = "p", out_col: str = "q_bh") -> pd.DataFrame:
    out = df.copy()
    mask = out[p_col].notna() & np.isfinite(out[p_col])
    out[out_col] = np.nan
    if mask.sum() > 0:
        out.loc[mask, out_col] = multipletests(out.loc[mask, p_col].astype(float).values, method="fdr_bh")[1]
    return out


def coef_table_from_result(result, model_name: str, outcome: str, cluster: Optional[str] = None) -> pd.DataFrame:
    names = list(getattr(result.model, "exog_names", []))
    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(result.bse, dtype=float)
    tvals = np.asarray(getattr(result, "tvalues", np.full_like(params, np.nan)), dtype=float)
    pvals = np.asarray(getattr(result, "pvalues", np.full_like(params, np.nan)), dtype=float)
    if len(names) != len(params):
        names = [f"b{i}" for i in range(len(params))]
    rows = []
    for name, est, se, t, p in zip(names, params, bse, tvals, pvals):
        rows.append({
            "model_name": model_name,
            "outcome": outcome,
            "coef": name,
            "est": float(est),
            "se": float(se),
            "stat": float(t),
            "p": float(p),
            "ci_lo": float(est - 1.96 * se),
            "ci_hi": float(est + 1.96 * se),
            "n_obs": int(getattr(result, "nobs", np.nan)) if np.isfinite(getattr(result, "nobs", np.nan)) else np.nan,
            "cluster": cluster or "",
        })
    return pd.DataFrame(rows)


def ols_cluster(formula: str, data: pd.DataFrame, cluster_col: str = "subject", model_name: str = "model") -> Tuple[object, pd.DataFrame]:
    """OLS with subject-cluster robust SE, aligned to rows retained by patsy.

    Statsmodels/patsy silently drop rows with missing values in variables used by
    the formula. Cluster labels must be indexed to that retained row set; passing
    the full data vector can produce misalignment when any model term has NA.
    """
    d = data.dropna(subset=[cluster_col]).copy()
    m = smf.ols(formula, data=d).fit()
    row_labels = m.model.data.row_labels
    groups = d.loc[row_labels, cluster_col]
    if len(groups) != int(m.nobs):
        raise RuntimeError(
            f"Cluster vector length mismatch for {model_name}: "
            f"groups={len(groups)} vs nobs={int(m.nobs)}"
        )
    r = m.get_robustcov_results(cov_type="cluster", groups=groups)
    outcome = formula.split("~", 1)[0].strip()
    return r, coef_table_from_result(r, model_name=model_name, outcome=outcome, cluster=cluster_col)


def gee_exchangeable(formula: str, data: pd.DataFrame, group_col: str = "subject") -> Dict:
    """Correct GEE implementation using patsy design matrices."""
    try:
        d = data.dropna(subset=[group_col]).copy()
        y, X = dmatrices(formula, data=d, return_type="dataframe")
        groups = d.loc[y.index, group_col].values
        model = GEE(y.iloc[:, 0], X, groups=groups, family=Gaussian(), cov_struct=Exchangeable())
        res = model.fit()
        return {
            "ok": True,
            "formula": formula,
            "params": {k: float(v) for k, v in zip(X.columns, res.params)},
            "bse": {k: float(v) for k, v in zip(X.columns, res.bse)},
            "pvalues": {k: float(v) for k, v in zip(X.columns, res.pvalues)},
            "n_obs": int(X.shape[0]),
            "n_groups": int(pd.Series(groups).nunique()),
        }
    except Exception as e:
        return {"ok": False, "formula": formula, "error": repr(e)}


def subject_one_sample(values: pd.Series, label: str, n_boot: int, n_perm: int, seed: int) -> Dict:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float).values
    out = {"test": label, "n_subjects": int(vals.size)}
    if vals.size < 2:
        out.update({"mean": np.nan, "se": np.nan, "t": np.nan, "p_t": np.nan, "p_signflip": np.nan, "ci_lo_boot": np.nan, "ci_hi_boot": np.nan, "dz": np.nan})
        return out
    tt = stats.ttest_1samp(vals, 0.0)
    ci_lo, ci_hi = mean_ci_boot(vals, n_boot=n_boot, seed=seed)
    out.update({
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)),
        "se": float(vals.std(ddof=1) / np.sqrt(vals.size)),
        "t": float(tt.statistic),
        "p_t": float(tt.pvalue),
        "p_signflip": sign_flip_p(vals, n_perm=n_perm, seed=seed),
        "ci_lo_boot": ci_lo,
        "ci_hi_boot": ci_hi,
        "dz": cohen_dz(vals),
    })
    try:
        wt = stats.wilcoxon(vals)
        out["wilcoxon_stat"] = float(wt.statistic)
        out["wilcoxon_p"] = float(wt.pvalue)
    except Exception as e:
        out["wilcoxon_error"] = repr(e)
    return out


def subject_two_group(values: pd.Series, group: pd.Series, label: str) -> Dict:
    d = pd.DataFrame({"value": values, "group": group}).dropna()
    out = {"test": label, "n": int(len(d)), "n_group0": int((d["group"] == 0).sum()), "n_group1": int((d["group"] == 1).sum())}
    if d["group"].nunique() != 2:
        out.update({"mean_group0": np.nan, "mean_group1": np.nan, "diff_group1_minus_group0": np.nan, "t": np.nan, "p_welch": np.nan})
        return out
    a = d.loc[d["group"] == 0, "value"].astype(float).values
    b = d.loc[d["group"] == 1, "value"].astype(float).values
    tt = stats.ttest_ind(b, a, equal_var=False)
    out.update({
        "mean_group0": float(a.mean()),
        "mean_group1": float(b.mean()),
        "diff_group1_minus_group0": float(b.mean() - a.mean()),
        "t": float(tt.statistic),
        "p_welch": float(tt.pvalue),
    })
    return out


# =============================================================================
# Data preparation
# =============================================================================

REQUIRED = [
    "subject", "run", "run_index", "trial_in_run", "global_trial_index",
    "condition", "temperature_raw", "temperature_z_subject",
    "rating_raw", "rating_z_subject", "up_first",
    "run_centered_subject", "trial_in_run_c", "global_trial_z_subject",
]


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Input is not the standardized master table. Missing columns: {missing}")

    df = df.copy()
    df["subject"] = df["subject"].astype(str)
    df["condition"] = df["condition"].astype(str).str.lower().str.strip()
    for col in [
        "run", "run_index", "trial_in_run", "global_trial_index", "temperature_raw",
        "temperature_z_subject", "rating_raw", "rating_z_subject", "up_first",
        "run_centered_subject", "trial_in_run_c", "global_trial_z_subject",
        "passive_context_shifted", "prev_rating_z_subject", "prev_temperature_subject",
        "prev_temp_minus_subject_mean", "has_prev_subject",
    ]:
        if col in df.columns:
            df[col] = finite_series(df[col])

    df = df.sort_values(["subject", "run_index", "trial_in_run"]).reset_index(drop=True)

    # Canonical task indicators
    df["is_passive"] = (df["condition"] == "passive").astype(int)
    df["is_reg"] = df["condition"].isin(["up", "down"]).astype(int)
    df["is_up"] = (df["condition"] == "up").astype(int)
    df["is_down"] = (df["condition"] == "down").astype(int)
    df["cond_ud_pm1"] = np.where(df["condition"] == "up", 1.0, np.where(df["condition"] == "down", -1.0, np.nan))

    # Stable history covariates.
    if "prev_rating_z_subject" not in df.columns or df["prev_rating_z_subject"].isna().all():
        df["prev_rating_z_subject"] = df.groupby("subject")["rating_z_subject"].shift(1)
    if "prev_temperature_z_subject" not in df.columns:
        df["prev_temperature_z_subject"] = df.groupby("subject")["temperature_z_subject"].shift(1)
    if "has_prev_subject" not in df.columns:
        df["has_prev_subject"] = df["prev_rating_z_subject"].notna().astype(int)

    if "passive_context_shifted" not in df.columns:
        df["passive_context_shifted"] = 0

    return df


# =============================================================================
# Passive baseline and latent variables
# =============================================================================


@dataclass
class PassiveModelSpec:
    name: str
    outcome: str
    formula: str


def fit_passive_models(df: pd.DataFrame) -> Dict[str, object]:
    passive = df[df["condition"] == "passive"].copy()
    specs = [
        PassiveModelSpec(
            name="raw_P0_sensory",
            outcome="rating_raw",
            formula="rating_raw ~ C(subject) + temperature_z_subject",
        ),
        PassiveModelSpec(
            name="raw_P1_time",
            outcome="rating_raw",
            formula="rating_raw ~ C(subject) + temperature_z_subject + trial_in_run_c + run_centered_subject + I(run_centered_subject**2) + passive_context_shifted",
        ),
        PassiveModelSpec(
            name="z_P0_sensory",
            outcome="rating_z_subject",
            formula="rating_z_subject ~ C(subject) + temperature_z_subject",
        ),
        PassiveModelSpec(
            name="z_P1_time",
            outcome="rating_z_subject",
            formula="rating_z_subject ~ C(subject) + temperature_z_subject + trial_in_run_c + run_centered_subject + I(run_centered_subject**2) + passive_context_shifted",
        ),
    ]
    models = {}
    for spec in specs:
        models[spec.name] = smf.ols(spec.formula, data=passive).fit()
    return models


def add_latent_variables(df: pd.DataFrame, models: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for name, model in models.items():
        pred_col = f"expected_pain_{name}"
        resid_col = f"prediction_error_{name}"
        out[pred_col] = model.predict(out)
        outcome = "rating_raw" if name.startswith("raw") else "rating_z_subject"
        out[resid_col] = out[outcome] - out[pred_col]

    # Primary latent variables for downstream analyses and fMRI linkage
    out["expected_pain_raw"] = out["expected_pain_raw_P1_time"]
    out["prediction_error_raw"] = out["prediction_error_raw_P1_time"]
    out["expected_pain_z"] = out["expected_pain_z_P1_time"]
    out["prediction_error_z"] = out["prediction_error_z_P1_time"]

    # Direction-corrected regulation success relative to passive expectation
    out["reg_success_raw"] = np.nan
    out.loc[out["condition"] == "up", "reg_success_raw"] = out.loc[out["condition"] == "up", "prediction_error_raw"]
    out.loc[out["condition"] == "down", "reg_success_raw"] = -out.loc[out["condition"] == "down", "prediction_error_raw"]

    out["reg_success_z"] = np.nan
    out.loc[out["condition"] == "up", "reg_success_z"] = out.loc[out["condition"] == "up", "prediction_error_z"]
    out.loc[out["condition"] == "down", "reg_success_z"] = -out.loc[out["condition"] == "down", "prediction_error_z"]

    # Absolute residual is not success; it captures deviation from passive expectation.
    out["abs_prediction_error_z"] = out["prediction_error_z"].abs()
    return out


# =============================================================================
# Analyses
# =============================================================================


def qc_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({"metric": "n_rows", "value": len(df)})
    rows.append({"metric": "n_subjects", "value": df["subject"].nunique()})
    rows.append({"metric": "n_runs", "value": df[["subject", "run"]].drop_duplicates().shape[0]})
    for cond in ["passive", "up", "down"]:
        rows.append({"metric": f"n_trials_{cond}", "value": int((df["condition"] == cond).sum())})
    rows.append({"metric": "n_rating_missing", "value": int(df["rating_raw"].isna().sum())})
    rows.append({"metric": "n_temperature_missing", "value": int(df["temperature_raw"].isna().sum())})
    rows.append({"metric": "rating_raw_min", "value": float(df["rating_raw"].min())})
    rows.append({"metric": "rating_raw_max", "value": float(df["rating_raw"].max())})
    rows.append({"metric": "n_rating_raw_gt_100", "value": int((df["rating_raw"] > 100).sum())})
    rows.append({"metric": "reg_runs", "value": ",".join(map(str, sorted(df.loc[df["is_reg"] == 1, "run"].dropna().unique().astype(int).tolist())))})
    rows.append({"metric": "n_subjects_up_first", "value": int(df.groupby("subject")["up_first"].first().sum())})
    rows.append({"metric": "n_subjects_down_first", "value": int((1 - df.groupby("subject")["up_first"].first()).sum())})
    return pd.DataFrame(rows)


def passive_model_summary(models: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        rows.append({
            "model_name": name,
            "formula": getattr(model.model, "formula", ""),
            "n_obs": int(model.nobs),
            "r2": float(getattr(model, "rsquared", np.nan)),
            "r2_adj": float(getattr(model, "rsquared_adj", np.nan)),
            "aic": float(model.aic),
            "bic": float(model.bic),
        })
    return pd.DataFrame(rows)


def run_task_validation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Main validation: all trials, passive as reference via explicit up/down dummies.
    formulas = {
        "V1_rating_z_condition_vs_passive": "rating_z_subject ~ is_up + is_down + temperature_z_subject + trial_in_run_c + run_centered_subject + passive_context_shifted + C(subject)",
        "V2_rating_raw_condition_vs_passive": "rating_raw ~ is_up + is_down + temperature_z_subject + trial_in_run_c + run_centered_subject + passive_context_shifted + C(subject)",
    }
    for name, formula in formulas.items():
        _, tbl = ols_cluster(formula, df, cluster_col="subject", model_name=name)
        rows.append(tbl)

    out = pd.concat(rows, ignore_index=True)
    keep = ["Intercept", "is_up", "is_down", "temperature_z_subject", "trial_in_run_c", "run_centered_subject", "passive_context_shifted"]
    return out[out["coef"].isin(keep)].reset_index(drop=True)


def subject_regulation_summary(df: pd.DataFrame) -> pd.DataFrame:
    reg = df[df["is_reg"] == 1].copy()
    # Subject means by condition for key behavioral variables.
    wide = reg.pivot_table(index="subject", columns="condition", values=[
        "rating_raw", "rating_z_subject", "prediction_error_raw", "prediction_error_z", "reg_success_raw", "reg_success_z",
        "temperature_raw", "temperature_z_subject",
    ], aggfunc="mean")
    wide.columns = [f"{var}_{cond}" for var, cond in wide.columns]
    wide = wide.reset_index()

    subj = df.groupby("subject", as_index=False).agg(
        sex=("sex", "first") if "sex" in df.columns else ("subject", "first"),
        age=("age", "first") if "age" in df.columns else ("run", "first"),
        up_first=("up_first", "first"),
        order_label=("order_label", "first") if "order_label" in df.columns else ("subject", "first"),
    )
    out = subj.merge(wide, on="subject", how="left")

    # Subject-specific passive encoding parameters. These are descriptive
    # individual-difference summaries, not used to build the passive expected-pain
    # model. They make subject-level figures non-circular and interpretable.
    param_rows = []
    for sid, dsub in df[df["condition"] == "passive"].groupby("subject", sort=False):
        row = {"subject": sid}
        row["passive_mean_rating_z"] = float(dsub["rating_z_subject"].mean())
        row["passive_mean_rating_raw"] = float(dsub["rating_raw"].mean())
        try:
            m = smf.ols(
                "rating_z_subject ~ temperature_z_subject + trial_in_run_c + run_centered_subject",
                data=dsub
            ).fit()
            row["passive_sensory_slope_z"] = float(m.params.get("temperature_z_subject", np.nan))
            row["passive_withinrun_slope_z"] = float(m.params.get("trial_in_run_c", np.nan))
            row["passive_acrossrun_slope_z"] = float(m.params.get("run_centered_subject", np.nan))
            row["passive_model_r2_subject"] = float(m.rsquared)
        except Exception:
            row["passive_sensory_slope_z"] = np.nan
            row["passive_withinrun_slope_z"] = np.nan
            row["passive_acrossrun_slope_z"] = np.nan
            row["passive_model_r2_subject"] = np.nan
        param_rows.append(row)
    passive_params = pd.DataFrame(param_rows)
    out = out.merge(passive_params, on="subject", how="left")

    # Direction-corrected success columns are condition-specific; down success lives in reg_success_z_down etc.
    out["success_asymmetry_z_up_minus_down"] = out.get("reg_success_z_up", np.nan) - out.get("reg_success_z_down", np.nan)
    out["success_asymmetry_raw_up_minus_down"] = out.get("reg_success_raw_up", np.nan) - out.get("reg_success_raw_down", np.nan)
    out["mean_reg_success_z"] = out[[c for c in ["reg_success_z_up", "reg_success_z_down"] if c in out.columns]].mean(axis=1)
    out["mean_reg_success_raw"] = out[[c for c in ["reg_success_raw_up", "reg_success_raw_down"] if c in out.columns]].mean(axis=1)

    # Task separation is defined as the up residual minus the down residual and is distinct from asymmetry.
    out["task_separation_prediction_error_z_up_minus_down"] = out.get("prediction_error_z_up", np.nan) - out.get("prediction_error_z_down", np.nan)
    out["task_separation_rating_z_up_minus_down"] = out.get("rating_z_subject_up", np.nan) - out.get("rating_z_subject_down", np.nan)
    out["task_separation_rating_raw_up_minus_down"] = out.get("rating_raw_up", np.nan) - out.get("rating_raw_down", np.nan)
    return out

def run_regulation_success_tests(df: pd.DataFrame, subj: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tests = []
    # Separate success relative to passive expectation.
    tests.append(subject_one_sample(subj["reg_success_z_up"], "up_success_z_gt_0", n_boot, n_perm, seed))
    tests.append(subject_one_sample(subj["reg_success_z_down"], "down_success_z_gt_0", n_boot, n_perm, seed + 1))
    tests.append(subject_one_sample(subj["mean_reg_success_z"], "mean_regulation_success_z_gt_0", n_boot, n_perm, seed + 2))
    tests.append(subject_one_sample(subj["reg_success_raw_up"], "up_success_raw_gt_0", n_boot, n_perm, seed + 3))
    tests.append(subject_one_sample(subj["reg_success_raw_down"], "down_success_raw_gt_0", n_boot, n_perm, seed + 4))

    # Targeted trial-level model for success > 0 with subject-cluster SE.
    reg = df[df["is_reg"] == 1].copy()
    _, tbl = ols_cluster("reg_success_z ~ 1", reg, "subject", "S_trial_mean_success_z_cluster")
    _, tbl_up = ols_cluster("prediction_error_z ~ 1", reg[reg["condition"] == "up"], "subject", "S_trial_up_prediction_error_z_cluster")
    down_regulation_trials = reg[reg["condition"] == "down"].copy()
    down_regulation_trials["down_success_z"] = -down_regulation_trials["prediction_error_z"]
    _, tbl_down = ols_cluster("down_success_z ~ 1", down_regulation_trials, "subject", "S_trial_down_success_z_cluster")
    trial_tbl = pd.concat([tbl, tbl_up, tbl_down], ignore_index=True)
    trial_tbl = trial_tbl[trial_tbl["coef"] == "Intercept"].reset_index(drop=True)
    return pd.DataFrame(tests), trial_tbl


def run_asymmetry_tests(df: pd.DataFrame, subj: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tests = []
    # The key asymmetry question: is direction-corrected success_up - success_down different from zero?
    tests.append(subject_one_sample(subj["success_asymmetry_z_up_minus_down"], "success_asymmetry_z_up_minus_down", n_boot, n_perm, seed))
    tests.append(subject_one_sample(subj["success_asymmetry_raw_up_minus_down"], "success_asymmetry_raw_up_minus_down", n_boot, n_perm, seed + 1))

    # Report task separation separately from direction-corrected success asymmetry.
    tests.append(subject_one_sample(subj["task_separation_prediction_error_z_up_minus_down"], "task_separation_residual_z_up_minus_down_NOT_asymmetry", n_boot, n_perm, seed + 2))
    tests.append(subject_one_sample(subj["task_separation_rating_z_up_minus_down"], "task_separation_rating_z_up_minus_down_NOT_asymmetry", n_boot, n_perm, seed + 3))

    reg = df[df["is_reg"] == 1].copy()
    # is_up coefficient = success_up - success_down, because outcome is direction-corrected success.
    formula = "reg_success_z ~ is_up + temperature_z_subject + trial_in_run_c + run_centered_subject + C(subject)"
    _, tbl = ols_cluster(formula, reg, "subject", "A_success_asymmetry_trial_cluster")
    tbl = tbl[tbl["coef"].isin(["is_up", "temperature_z_subject", "trial_in_run_c", "run_centered_subject"])]
    return pd.DataFrame(tests), tbl.reset_index(drop=True)


def run_habituation_tests(df: pd.DataFrame) -> pd.DataFrame:
    models = []
    # Within-run changes in pain reports.
    within_specs = {
        "H1_within_run_passive_rating_z": "rating_z_subject ~ temperature_z_subject + trial_in_run_c + run_centered_subject + passive_context_shifted + C(subject)",
        "H2_within_run_all_rating_z": "rating_z_subject ~ temperature_z_subject + is_up + is_down + trial_in_run_c + run_centered_subject + passive_context_shifted + C(subject)",
    }
    for name, formula in within_specs.items():
        data = df[df["condition"] == "passive"].copy() if "passive" in name else df.copy()
        _, tbl = ols_cluster(formula, data, "subject", name)
        models.append(tbl[tbl["coef"].isin(["trial_in_run_c", "temperature_z_subject", "is_up", "is_down"])])

    # Within-run changes in regulation success, plus asymmetry in that change.
    reg = df[df["is_reg"] == 1].copy()
    formula_success = "reg_success_z ~ trial_in_run_c + is_up + is_up:trial_in_run_c + temperature_z_subject + run_centered_subject + C(subject)"
    _, tbl_success = ols_cluster(formula_success, reg, "subject", "H3_within_run_reg_success_z")
    models.append(tbl_success[tbl_success["coef"].isin(["trial_in_run_c", "is_up", "is_up:trial_in_run_c", "temperature_z_subject"])])

    # Across-run/session drift. Passive-only is primary; all-trials is robustness.
    across_specs = {
        "H4_across_run_passive_rating_z": "rating_z_subject ~ temperature_z_subject + run_centered_subject + I(run_centered_subject**2) + trial_in_run_c + passive_context_shifted + C(subject)",
        "H5_across_run_all_rating_z_sensitivity": "rating_z_subject ~ temperature_z_subject + is_up + is_down + run_centered_subject + I(run_centered_subject**2) + trial_in_run_c + passive_context_shifted + C(subject)",
    }
    for name, formula in across_specs.items():
        data = df[df["condition"] == "passive"].copy() if "passive" in name else df.copy()
        _, tbl = ols_cluster(formula, data, "subject", name)
        models.append(tbl[tbl["coef"].isin(["run_centered_subject", "I(run_centered_subject ** 2)", "temperature_z_subject", "is_up", "is_down"])])

    out = pd.concat(models, ignore_index=True)
    # FDR only across the pre-specified time terms, not all nuisance covariates.
    mask = out["coef"].isin(["trial_in_run_c", "is_up:trial_in_run_c", "run_centered_subject", "I(run_centered_subject ** 2)"])
    out.loc[:, "q_bh_time_terms"] = np.nan
    if mask.sum() > 0:
        out.loc[mask, "q_bh_time_terms"] = multipletests(out.loc[mask, "p"].values, method="fdr_bh")[1]
    return out.reset_index(drop=True)


def run_order_tests(subj: pd.DataFrame) -> pd.DataFrame:
    tests = []
    # Subject-level only. up_first = 1 vs down_first = 0.
    targets = [
        ("order_effect_on_mean_success_z", "mean_reg_success_z"),
        ("order_effect_on_success_asymmetry_z", "success_asymmetry_z_up_minus_down"),
        ("order_effect_on_up_success_z", "reg_success_z_up"),
        ("order_effect_on_down_success_z", "reg_success_z_down"),
        ("order_effect_on_task_separation_rating_z", "task_separation_rating_z_up_minus_down"),
        ("order_effect_on_task_separation_residual_z", "task_separation_prediction_error_z_up_minus_down"),
    ]
    for label, col in targets:
        if col in subj.columns:
            tests.append(subject_two_group(subj[col], subj["up_first"], label))
    out = pd.DataFrame(tests)
    return bh_fdr_table(out.rename(columns={"p_welch": "p"}), p_col="p", out_col="q_bh_order")


def model_formulas(include_subject_fe: bool = True) -> Dict[str, str]:
    """Nested behavioral mechanism models.

    M0: sensory/time only
    M1: additive regulation shifts
    M2: regulation changes sensory gain
    M3: additive regulation + history/state
    M4: history/state-dependent regulation
    """
    subj = " + C(subject)" if include_subject_fe else ""
    return {
        "M0_sensory_only": f"rating_z_subject ~ temperature_z_subject + trial_in_run_c + run_centered_subject + passive_context_shifted{subj}",
        "M1_additive_regulatory_shift": f"rating_z_subject ~ temperature_z_subject + is_up + is_down + trial_in_run_c + run_centered_subject + passive_context_shifted{subj}",
        "M2_gain_modulation": f"rating_z_subject ~ temperature_z_subject + is_up + is_down + is_up:temperature_z_subject + is_down:temperature_z_subject + trial_in_run_c + run_centered_subject + passive_context_shifted{subj}",
        "M3_history_state": f"rating_z_subject ~ temperature_z_subject + is_up + is_down + prev_rating_z_subject + prev_temperature_z_subject + trial_in_run_c + run_centered_subject + passive_context_shifted{subj}",
        "M4_state_dependent_regulation": f"rating_z_subject ~ temperature_z_subject + is_up + is_down + prev_rating_z_subject + prev_temperature_z_subject + is_up:prev_rating_z_subject + is_down:prev_rating_z_subject + trial_in_run_c + run_centered_subject + passive_context_shifted{subj}",
    }


def nested_lrt_row(models: Dict[str, object], restricted: str, full: str) -> Dict:
    mr = models[restricted]
    mf = models[full]
    df_diff = float(mf.df_model - mr.df_model)
    lr = float(2 * (mf.llf - mr.llf))
    p = float(stats.chi2.sf(lr, df_diff)) if df_diff > 0 else np.nan
    return {
        "comparison": f"{restricted} -> {full}",
        "restricted_model": restricted,
        "full_model": full,
        "df_diff": df_diff,
        "lr_chi2": lr,
        "p_lrt": p,
        "delta_aic_full_minus_restricted": float(mf.aic - mr.aic),
        "delta_bic_full_minus_restricted": float(mf.bic - mr.bic),
    }


def run_loso_cv_model_comparison(df_common: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-subject-out prediction using models without subject fixed effects.

    The AIC/BIC comparison uses C(subject) to remove stable between-person
    offsets. For out-of-subject prediction, C(subject) cannot be used because the
    held-out subject has an unseen category. Therefore this CV table uses the
    same predictors but no subject fixed effects. It is a complementary
    generalization check, not a replacement for the within-subject AIC/BIC table.
    """
    formulas = model_formulas(include_subject_fe=False)
    rows = []
    for name, formula in formulas.items():
        preds = []
        truths = []
        n_failed = 0
        for sid in df_common["subject"].dropna().unique():
            train = df_common[df_common["subject"] != sid].copy()
            test = df_common[df_common["subject"] == sid].copy()
            try:
                m = smf.ols(formula, data=train).fit()
                yhat = np.asarray(m.predict(test), dtype=float)
                y = test["rating_z_subject"].to_numpy(dtype=float)
                preds.append(yhat)
                truths.append(y)
            except Exception:
                n_failed += 1
                continue
        if preds:
            pred = np.concatenate(preds)
            y = np.concatenate(truths)
            err = y - pred
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum(y ** 2))
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        else:
            rmse, mae, r2 = np.nan, np.nan, np.nan
        rows.append({
            "model_name": name,
            "formula_no_subject_fe": formula,
            "cv_scheme": "leave_one_subject_out_no_subject_FE",
            "n_obs_tested": int(sum(len(x) for x in truths)) if truths else 0,
            "n_subjects_failed": int(n_failed),
            "rmse": rmse,
            "mae": mae,
            "r2_vs_zero": r2,
        })
    out = pd.DataFrame(rows)
    out["delta_rmse"] = out["rmse"] - out["rmse"].min()
    return out.sort_values("rmse").reset_index(drop=True)


def run_model_comparison(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    # Main behavioral mechanism comparison: additive shift vs sensory gain vs history/state.
    # All AIC/BIC and likelihood-ratio comparisons use the same complete-case rows.
    # This avoids invalid comparisons between models with different N.
    formulas = model_formulas(include_subject_fe=True)
    common = df.dropna(subset=["prev_rating_z_subject", "prev_temperature_z_subject"]).copy()
    common_n = int(common.shape[0])

    rows = []
    fitted = {}
    gee = {}
    for name, formula in formulas.items():
        m = smf.ols(formula, data=common).fit()
        fitted[name] = m
        rows.append({
            "model_name": name,
            "formula": formula,
            "comparison_sample": "common_history_complete_cases",
            "n_obs": int(m.nobs),
            "n_common_rows": common_n,
            "r2": float(m.rsquared),
            "r2_adj": float(m.rsquared_adj),
            "loglik": float(m.llf),
            "df_model": float(m.df_model),
            "aic": float(m.aic),
            "bic": float(m.bic),
            "delta_aic": np.nan,
            "delta_bic": np.nan,
        })
        if name in ["M1_additive_regulatory_shift", "M2_gain_modulation", "M3_history_state", "M4_state_dependent_regulation"]:
            gee[name] = gee_exchangeable(formula, data=common, group_col="subject")

    comp = pd.DataFrame(rows)
    comp["delta_aic"] = comp["aic"] - comp["aic"].min()
    comp["delta_bic"] = comp["bic"] - comp["bic"].min()
    comp = comp.sort_values("aic").reset_index(drop=True)

    nested = pd.DataFrame([
        nested_lrt_row(fitted, "M0_sensory_only", "M1_additive_regulatory_shift"),
        nested_lrt_row(fitted, "M1_additive_regulatory_shift", "M2_gain_modulation"),
        nested_lrt_row(fitted, "M1_additive_regulatory_shift", "M3_history_state"),
        nested_lrt_row(fitted, "M3_history_state", "M4_state_dependent_regulation"),
    ])
    nested = bh_fdr_table(nested.rename(columns={"p_lrt": "p"}), p_col="p", out_col="q_bh_lrt")
    nested = nested.rename(columns={"p": "p_lrt"})

    cv = run_loso_cv_model_comparison(common)
    return comp, nested, cv, gee

def run_precision_tests(df: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Does regulation change residual dispersion/precision? This is exploratory.
    # Fit an additive model, then compare subject-level residual SD by condition.
    m = smf.ols(
        "rating_z_subject ~ temperature_z_subject + is_up + is_down + trial_in_run_c + run_centered_subject + passive_context_shifted + C(subject)",
        data=df
    ).fit()
    d = df.copy()
    d["additive_model_residual_z"] = m.resid
    subj_sd = d.groupby(["subject", "condition"], as_index=False)["additive_model_residual_z"].std()
    wide = subj_sd.pivot(index="subject", columns="condition", values="additive_model_residual_z").reset_index()
    wide["sd_reg_mean"] = wide[[c for c in ["up", "down"] if c in wide.columns]].mean(axis=1)
    wide["sd_up_minus_down"] = wide.get("up", np.nan) - wide.get("down", np.nan)
    wide["sd_reg_minus_passive"] = wide["sd_reg_mean"] - wide.get("passive", np.nan)

    tests = []
    tests.append(subject_one_sample(wide["sd_up_minus_down"], "precision_sd_up_minus_down", n_boot, n_perm, seed))
    tests.append(subject_one_sample(wide["sd_reg_minus_passive"], "precision_sd_reg_mean_minus_passive", n_boot, n_perm, seed + 1))
    return wide, pd.DataFrame(tests)


# =============================================================================
# Figures
# =============================================================================


def fig_condition_means(df: pd.DataFrame, path: Path) -> None:
    sub = df.groupby(["subject", "condition"], as_index=False).agg(
        rating_z=("rating_z_subject", "mean"),
        rating_raw=("rating_raw", "mean"),
        pred_err_z=("prediction_error_z", "mean"),
    )
    order = ["passive", "down", "up"]
    g = sub.groupby("condition")["rating_z"].agg(["mean", "std", "count"]).reindex(order).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"])
    x = np.arange(len(g))
    plt.figure(figsize=(6, 4))
    plt.errorbar(x, g["mean"], yerr=1.96 * g["se"], fmt="o")
    plt.xticks(x, g["condition"])
    plt.ylabel("Pain rating, subject-z")
    plt.title("Behavioral validation: condition means")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def fig_temperature_curves(df: pd.DataFrame, path: Path) -> None:
    # Subject means by rounded temperature and condition.
    d = df.copy()
    d["temperature_bin"] = d["temperature_raw"].round(1)
    sub = d.groupby(["subject", "condition", "temperature_bin"], as_index=False)["rating_z_subject"].mean()
    g = sub.groupby(["condition", "temperature_bin"], as_index=False).agg(mean=("rating_z_subject", "mean"), se=("rating_z_subject", lambda x: x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.nan))
    plt.figure(figsize=(7, 4.5))
    for cond in ["passive", "down", "up"]:
        gg = g[g["condition"] == cond]
        if len(gg) == 0:
            continue
        plt.errorbar(gg["temperature_bin"], gg["mean"], yerr=1.96 * gg["se"], marker="o", label=cond)
    plt.xlabel("Recorded stimulus intensity / temperature")
    plt.ylabel("Pain rating, subject-z")
    plt.title("Temperature-pain function by condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def fig_success_by_condition(df: pd.DataFrame, path: Path) -> None:
    reg = df[df["is_reg"] == 1].copy()
    sub = reg.groupby(["subject", "condition"], as_index=False)["reg_success_z"].mean()
    g = sub.groupby("condition")["reg_success_z"].agg(["mean", "std", "count"]).reindex(["down", "up"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"])
    x = np.arange(len(g))
    plt.figure(figsize=(6, 4))
    plt.errorbar(x, g["mean"], yerr=1.96 * g["se"], fmt="o")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, g["condition"])
    plt.ylabel("Direction-corrected regulation success, z")
    plt.title("Regulation success relative to passive expectation")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def fig_within_run_success(df: pd.DataFrame, path: Path) -> None:
    reg = df[df["is_reg"] == 1].copy()
    sub = reg.groupby(["subject", "condition", "trial_in_run"], as_index=False)["reg_success_z"].mean()
    g = sub.groupby(["condition", "trial_in_run"], as_index=False).agg(mean=("reg_success_z", "mean"), se=("reg_success_z", lambda x: x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.nan))
    plt.figure(figsize=(7, 4))
    for cond in ["down", "up"]:
        gg = g[g["condition"] == cond]
        plt.errorbar(gg["trial_in_run"], gg["mean"], yerr=1.96 * gg["se"], marker="o", label=cond)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Trial within regulation run")
    plt.ylabel("Regulation success, z")
    plt.title("Within-run dynamics of regulation success")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def fig_across_run_passive(df: pd.DataFrame, path: Path) -> None:
    passive = df[df["condition"] == "passive"].copy()
    sub = passive.groupby(["subject", "run"], as_index=False).agg(rating_z=("rating_z_subject", "mean"), temp=("temperature_z_subject", "mean"))
    g = sub.groupby("run")["rating_z"].agg(["mean", "std", "count"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"])
    plt.figure(figsize=(7, 4))
    plt.errorbar(g["run"], g["mean"], yerr=1.96 * g["se"], marker="o")
    plt.xlabel("Run")
    plt.ylabel("Passive pain rating, subject-z")
    plt.title("Across-run passive rating drift")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def fig_up_down_success_space(subj: pd.DataFrame, path: Path) -> None:
    """Non-circular subject-level success figure: up success vs down success."""
    d = subj[["reg_success_z_up", "reg_success_z_down"]].dropna().copy()
    x = d["reg_success_z_down"].to_numpy(dtype=float)
    y = d["reg_success_z_up"].to_numpy(dtype=float)
    r = np.nan
    p = np.nan
    if len(d) >= 3:
        r, p = stats.pearsonr(x, y)
    lo = float(np.nanmin([x.min(), y.min(), 0]))
    hi = float(np.nanmax([x.max(), y.max(), 0]))
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1)
    plt.axhline(0, linestyle=":", linewidth=1)
    plt.axvline(0, linestyle=":", linewidth=1)
    plt.xlabel("Down-regulation success, z")
    plt.ylabel("Up-regulation success, z")
    title = "Subject-level regulation success: up vs down"
    if np.isfinite(r):
        title += f"\nr={r:.2f}, p={p:.3g}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def fig_sensory_sensitivity_vs_success(subj: pd.DataFrame, path: Path) -> None:
    """Independent passive sensory sensitivity vs regulation capacity."""
    d = subj[["passive_sensory_slope_z", "mean_reg_success_z", "up_first"]].dropna().copy()
    x = d["passive_sensory_slope_z"].to_numpy(dtype=float)
    y = d["mean_reg_success_z"].to_numpy(dtype=float)
    r = np.nan
    p = np.nan
    if len(d) >= 3:
        r, p = stats.pearsonr(x, y)
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Passive sensory sensitivity slope, z units")
    plt.ylabel("Mean regulation success, z")
    title = "Passive sensory sensitivity vs regulation capacity"
    if np.isfinite(r):
        title += f"\nr={r:.2f}, p={p:.3g}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


# =============================================================================
# Report
# =============================================================================


def find_row(tbl: pd.DataFrame, model: str, coef: str) -> Optional[pd.Series]:
    d = tbl[(tbl["model_name"] == model) & (tbl["coef"] == coef)]
    if len(d) == 0:
        return None
    return d.iloc[0]


def ptxt(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def write_report(
    outdir: Path,
    manifest: Dict,
    qc: pd.DataFrame,
    task_val: pd.DataFrame,
    success_subject: pd.DataFrame,
    asym_subject: pd.DataFrame,
    asym_trial: pd.DataFrame,
    habituation: pd.DataFrame,
    order_tests: pd.DataFrame,
    model_comp: pd.DataFrame,
    model_nested: pd.DataFrame,
    model_cv: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# ds000140 behavioral computational analysis \n")
    lines.append("\n")
    lines.append("## Input and reproducibility\n")
    lines.append(f"- Generated: {manifest['timestamp_local']}\n")
    lines.append(f"- Input: `{manifest['input_path']}`\n")
    lines.append(f"- SHA256: `{manifest['input_sha256']}`\n")
    lines.append(f"- Python: `{manifest['python_version']}`\n")
    lines.append("\n")

    q = dict(zip(qc["metric"], qc["value"]))
    lines.append("## Dataset summary\n")
    lines.append(f"- Subjects: {q.get('n_subjects')}\n")
    lines.append(f"- Trials: {q.get('n_rows')} total; passive={q.get('n_trials_passive')}, up={q.get('n_trials_up')}, down={q.get('n_trials_down')}\n")
    lines.append(f"- Regulation runs: {q.get('reg_runs')}\n")
    lines.append(f"- Order balance: up-first={q.get('n_subjects_up_first')}, down-first={q.get('n_subjects_down_first')}\n")
    lines.append(f"- rating_raw range: {q.get('rating_raw_min')} to {q.get('rating_raw_max')}; values >100: {q.get('n_rating_raw_gt_100')}\n")
    lines.append("\n")

    lines.append("## 1. Task validation: does explicit regulation change pain reports relative to passive?\n")
    up = find_row(task_val, "V1_rating_z_condition_vs_passive", "is_up")
    down = find_row(task_val, "V1_rating_z_condition_vs_passive", "is_down")
    temp = find_row(task_val, "V1_rating_z_condition_vs_passive", "temperature_z_subject")
    if up is not None:
        lines.append(f"- Up vs passive, rating_z: β={up.est:.4f}, SE={up.se:.4f}, p={ptxt(up.p)}.\n")
    if down is not None:
        lines.append(f"- Down vs passive, rating_z: β={down.est:.4f}, SE={down.se:.4f}, p={ptxt(down.p)}.\n")
    if temp is not None:
        lines.append(f"- Temperature effect, rating_z: β={temp.est:.4f}, SE={temp.se:.4f}, p={ptxt(temp.p)}.\n")
    lines.append("\n")

    lines.append("## 2. Regulation success and direction-corrected behavioral asymmetry\n")
    for _, r in success_subject.iterrows():
        if r["test"] in ["up_success_z_gt_0", "down_success_z_gt_0", "mean_regulation_success_z_gt_0"]:
            lines.append(f"- {r['test']}: mean={r['mean']:.4f}, dz={r['dz']:.3f}, p_t={ptxt(r['p_t'])}, sign-flip p={ptxt(r['p_signflip'])}.\n")
    for _, r in asym_subject.iterrows():
        if r["test"] == "success_asymmetry_z_up_minus_down":
            lines.append(f"- True success asymmetry, success_up − success_down: mean={r['mean']:.4f}, dz={r['dz']:.3f}, p_t={ptxt(r['p_t'])}, sign-flip p={ptxt(r['p_signflip'])}.\n")
    ar = find_row(asym_trial, "A_success_asymmetry_trial_cluster", "is_up")
    if ar is not None:
        lines.append(f"- Trial-level success asymmetry model, is_up coefficient: β={ar.est:.4f}, SE={ar.se:.4f}, p={ptxt(ar.p)}.\n")
    lines.append("\n")
    lines.append("Important: up-minus-down residual/rating separation is reported separately and is not labeled as success asymmetry. True asymmetry is tested on direction-corrected success.\n")
    lines.append("\n")

    lines.append("## 3–4. Within-run and across-run habituation/sensitization\n")
    for model, coef, label in [
        ("H1_within_run_passive_rating_z", "trial_in_run_c", "Passive within-run rating drift"),
        ("H3_within_run_reg_success_z", "trial_in_run_c", "Within-run regulation-success drift"),
        ("H3_within_run_reg_success_z", "is_up:trial_in_run_c", "Up/down difference in success drift"),
        ("H4_across_run_passive_rating_z", "run_centered_subject", "Passive across-run linear drift"),
        ("H4_across_run_passive_rating_z", "I(run_centered_subject ** 2)", "Passive across-run quadratic drift"),
    ]:
        rr = find_row(habituation, model, coef)
        if rr is not None:
            q = rr.get("q_bh_time_terms", np.nan)
            q_part = f", q={ptxt(q)}" if np.isfinite(q) else ""
            lines.append(f"- {label}: β={rr.est:.4f}, SE={rr.se:.4f}, p={ptxt(rr.p)}{q_part}.\n")
    lines.append("\n")

    lines.append("## 5. Regulation order: up-first vs down-first\n")
    for _, r in order_tests.iterrows():
        if r["test"] in ["order_effect_on_mean_success_z", "order_effect_on_success_asymmetry_z", "order_effect_on_task_separation_rating_z"]:
            lines.append(f"- {r['test']}: group difference up_first−down_first={r['diff_group1_minus_group0']:.4f}, p={ptxt(r['p'])}, q={ptxt(r['q_bh_order'])}.\n")
    lines.append("\n")
    lines.append("Order tests are subject-level Welch tests. The problematic trial-level condition × order model is intentionally omitted because condition, run, and order are structurally coupled in this design.\n")
    lines.append("\n")

    lines.append("## Computational model comparison\n")
    best_aic = model_comp.sort_values("aic").iloc[0]
    best_bic = model_comp.sort_values("bic").iloc[0]
    lines.append(f"- AIC/BIC comparisons were computed on the same complete-case trial set (n={int(best_aic['n_obs'])}) to avoid unequal-N information-criterion artifacts.\n")
    lines.append(f"- Best AIC model: {best_aic['model_name']} (ΔAIC=0, ΔBIC={best_aic['delta_bic']:.2f}).\n")
    lines.append(f"- Best BIC model: {best_bic['model_name']} (ΔBIC=0, ΔAIC={best_bic['delta_aic']:.2f}).\n")
    for comp_name in ["M0_sensory_only -> M1_additive_regulatory_shift", "M1_additive_regulatory_shift -> M2_gain_modulation", "M1_additive_regulatory_shift -> M3_history_state", "M3_history_state -> M4_state_dependent_regulation"]:
        rr = model_nested[model_nested["comparison"] == comp_name]
        if len(rr):
            r = rr.iloc[0]
            lines.append(f"- Nested LRT {comp_name}: χ²={r['lr_chi2']:.2f}, df={r['df_diff']:.0f}, p={ptxt(r['p_lrt'])}, q={ptxt(r['q_bh_lrt'])}; ΔBIC(full−restricted)={r['delta_bic_full_minus_restricted']:.2f}.\n")
    if len(model_cv):
        cvbest = model_cv.sort_values("rmse").iloc[0]
        lines.append(f"- LOSO predictive check without subject fixed effects: best RMSE model was {cvbest['model_name']} (RMSE={cvbest['rmse']:.4f}). This is complementary and not used as the primary inferential test.\n")
    lines.append("- These models compare additive regulatory shift, sensory-gain modulation, and history/state terms. Treat model comparison as a behavioral decomposition that generates regressors for later fMRI analyses, not as causal proof by itself.\n")
    lines.append("\n")

    lines.append("## Key output files\n")
    lines.append("- `tables/trialwise_behavior_latents.tsv`: expected pain, prediction error, and regulation-success variables for fMRI linkage.\n")
    lines.append("- `tables/subject_level_behavior_parameters.tsv`: subject-level sensitivity/success/asymmetry/order metrics.\n")
    lines.append("- `tables/task_validation_condition_effects.tsv`: regulation vs passive validation.\n")
    lines.append("- `tables/regulation_success_subject_tests.tsv` and `tables/success_asymmetry_subject_tests.tsv`: core subject-level behavioral inference.\n")
    lines.append("- `tables/habituation_time_effects.tsv`: within-run and across-run drift tests.\n")
    lines.append("- `tables/order_effect_subject_tests.tsv`: order analyses.\n")
    lines.append("- `tables/computational_model_comparison.tsv`: common-N AIC/BIC model comparison.\n")
    lines.append("- `tables/computational_model_nested_tests.tsv`: likelihood-ratio tests for nested mechanism models.\n")
    lines.append("- `tables/computational_model_loso_cv.tsv`: leave-one-subject-out predictive sensitivity check.\n")
    lines.append("- `figures/`: QC and summary figures.\n")

    (outdir / "behavioral_computational_report.md").write_text("".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavioral computational analysis pipeline for ds000140.")
    parser.add_argument("--data", required=True, type=str, help="Path to behavior_trials_master_clean.tsv")
    parser.add_argument("--outdir", required=True, type=str, help="Output directory")
    parser.add_argument("--boot-R", default=5000, type=int, help="Bootstrap iterations for mean CIs")
    parser.add_argument("--perm-R", default=20000, type=int, help="Sign-flip permutations")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--validation-run", action="store_true", help="Use fewer resamples for validation")
    args = parser.parse_args()

    if args.validation_run:
        args.boot_R = min(args.boot_R, 500)
        args.perm_R = min(args.perm_R, 2000)

    data_path = Path(args.data).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    tables = outdir / "tables"
    figs = outdir / "figures"
    ensure_dir(outdir)
    ensure_dir(tables)
    ensure_dir(figs)

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df = load_and_prepare(data_path)
    passive_models = fit_passive_models(df)
    df = add_latent_variables(df, passive_models)

    # Core tables
    qc = qc_summary(df)
    save_tsv(qc, tables / "data_qc_summary.tsv")

    passive_summary = passive_model_summary(passive_models)
    save_tsv(passive_summary, tables / "passive_baseline_model_summary.tsv")

    # Save latent table for fMRI linkage.
    latent_cols = [
        "subject", "run", "run_index", "trial_in_run", "global_trial_index", "trial_id",
        "condition", "is_reg", "is_up", "is_down", "up_first", "order_label",
        "temperature_raw", "temperature_z_subject", "rating_raw", "rating_z_subject",
        "expected_pain_raw", "prediction_error_raw", "expected_pain_z", "prediction_error_z",
        "reg_success_raw", "reg_success_z", "abs_prediction_error_z",
        "trial_in_run_c", "run_centered_subject", "global_trial_z_subject",
        "prev_rating_z_subject", "prev_temperature_z_subject",
        "stim_onset_runrel", "rating_onset_runrel", "stim_duration", "rating_duration",
    ]
    latent_cols = [c for c in latent_cols if c in df.columns]
    save_tsv(df[latent_cols], tables / "trialwise_behavior_latents.tsv")

    # Analyses
    task_val = run_task_validation(df)
    save_tsv(task_val, tables / "task_validation_condition_effects.tsv")

    subj = subject_regulation_summary(df)
    save_tsv(subj, tables / "subject_level_behavior_parameters.tsv")

    success_subject, success_trial = run_regulation_success_tests(df, subj, args.boot_R, args.perm_R, args.seed)
    save_tsv(success_subject, tables / "regulation_success_subject_tests.tsv")
    save_tsv(success_trial, tables / "regulation_success_trial_cluster_tests.tsv")

    asym_subject, asym_trial = run_asymmetry_tests(df, subj, args.boot_R, args.perm_R, args.seed + 100)
    save_tsv(asym_subject, tables / "success_asymmetry_subject_tests.tsv")
    save_tsv(asym_trial, tables / "success_asymmetry_trial_cluster_model.tsv")

    habituation = run_habituation_tests(df)
    save_tsv(habituation, tables / "habituation_time_effects.tsv")

    order_tests = run_order_tests(subj)
    save_tsv(order_tests, tables / "order_effect_subject_tests.tsv")

    model_comp, model_nested, model_cv, gee_sensitivity = run_model_comparison(df)
    save_tsv(model_comp, tables / "computational_model_comparison.tsv")
    save_tsv(model_nested, tables / "computational_model_nested_tests.tsv")
    save_tsv(model_cv, tables / "computational_model_loso_cv.tsv")
    save_json(gee_sensitivity, tables / "gee_sensitivity_models.json")

    precision_subj, precision_tests = run_precision_tests(df, args.boot_R, args.perm_R, args.seed + 200)
    save_tsv(precision_subj, tables / "precision_subject_condition_sd.tsv")
    save_tsv(precision_tests, tables / "precision_tests.tsv")

    # Figures
    fig_condition_means(df, figs / "fig_01_condition_means_rating_z.png")
    fig_temperature_curves(df, figs / "fig_02_temperature_curves_by_condition.png")
    fig_success_by_condition(df, figs / "fig_03_regulation_success_by_condition.png")
    fig_within_run_success(df, figs / "fig_04_within_run_success.png")
    fig_across_run_passive(df, figs / "fig_05_across_run_passive_rating.png")
    fig_up_down_success_space(subj, figs / "fig_06_up_vs_down_success_subjects.png")
    fig_sensory_sensitivity_vs_success(subj, figs / "fig_07_passive_sensitivity_vs_success.png")

    # Manifest/report
    manifest = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(data_path),
        "input_sha256": sha256_file(data_path),
        "script": Path(__file__).name,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "boot_R": int(args.boot_R),
        "perm_R": int(args.perm_R),
        "seed": int(args.seed),
        "primary_outcome": "rating_z_subject and direction-corrected reg_success_z",
        "primary_latent_model": "Passive expected pain: rating_z_subject ~ C(subject)+temperature_z_subject+trial/run time covariates+passive_context_shifted, fitted on passive trials only",
        "model_comparison_note": "AIC/BIC and nested LRTs use common complete-case rows for all mechanism models.",
    }
    save_json(manifest, outdir / "manifest.json")

    write_report(
        outdir=outdir,
        manifest=manifest,
        qc=qc,
        task_val=task_val,
        success_subject=success_subject,
        asym_subject=asym_subject,
        asym_trial=asym_trial,
        habituation=habituation,
        order_tests=order_tests,
        model_comp=model_comp,
        model_nested=model_nested,
        model_cv=model_cv,
    )

    print("Processing completed successfully.")
    print(f"Output directory: {outdir}")
    print(f"Report: {outdir / 'behavioral_computational_report.md'}")


if __name__ == "__main__":
    main()
