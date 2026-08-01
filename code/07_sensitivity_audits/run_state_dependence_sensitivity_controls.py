#!/usr/bin/env python3
"""State-dependence sensitivity controls.

The workflow evaluates whether state-dependent regulation findings are robust
to bounded rating scales, passive-baseline coupling, available response range,
and stimulus-intensity stratification. Input tables may be supplied explicitly;
repository-relative behavioral tables are used where compatible.

Primary outputs include model terms and contrasts, cross-fitted passive-baseline
results, stimulus and expected-pain bin analyses, and descriptive margins.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------
# Utility functions
# -----------------------------


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def require_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}\nAvailable columns: {list(df.columns)}")


def choose_col(df: pd.DataFrame, candidates: list[str], label: str, required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not find {label}. Tried: {candidates}\nAvailable columns: {list(df.columns)}")
    return None


def center(df: pd.DataFrame, col: str, new_col: Optional[str] = None) -> str:
    if new_col is None:
        new_col = f"{col}_c"
    df[new_col] = df[col] - df[col].mean(skipna=True)
    return new_col


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean()) / sd


def bh_fdr(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce")
    ok = p.notna() & np.isfinite(p)
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    if ok.sum() > 0:
        out.loc[ok] = multipletests(p.loc[ok].values, method="fdr_bh")[1]
    return out


def safe_qcut(s: pd.Series, q: int, labels: list[str]) -> pd.Series:
    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop")
    except Exception:
        # Use rank-based quantile assignment when tied values prevent direct binning.
        return pd.qcut(s.rank(method="first"), q=q, labels=labels, duplicates="drop")


def fit_cluster_ols(df: pd.DataFrame, formula: str, cluster_col: str = "subject"):
    d = df.copy()
    needed = list(set([cluster_col]))
    d = d.dropna(subset=needed)
    model = smf.ols(formula, data=d, missing="drop")
    row_labels = model.data.row_labels
    groups = d.loc[row_labels, cluster_col]
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return fit


def fit_fractional_logit(df: pd.DataFrame, formula: str, cluster_col: str = "subject"):
    d = df.copy().dropna(subset=[cluster_col])
    model = smf.glm(formula, data=d, family=sm.families.Binomial(), missing="drop")
    row_labels = model.data.row_labels
    groups = d.loc[row_labels, cluster_col]
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return fit


def extract_terms(fit, model_name: str, terms: Optional[list[str]] = None, extra: Optional[dict] = None) -> pd.DataFrame:
    if terms is None:
        terms = list(fit.params.index)
    rows = []
    for term in terms:
        if term not in fit.params.index:
            continue
        rows.append({
            "model": model_name,
            "term": term,
            "beta": float(fit.params[term]),
            "se": float(fit.bse[term]) if term in fit.bse.index else np.nan,
            "t_or_z": float(fit.tvalues[term]) if term in fit.tvalues.index else np.nan,
            "p": float(fit.pvalues[term]) if term in fit.pvalues.index else np.nan,
            "n": int(getattr(fit, "nobs", np.nan)),
            "r2": float(getattr(fit, "rsquared", np.nan)) if hasattr(fit, "rsquared") else np.nan,
        })
    out = pd.DataFrame(rows)
    if extra:
        for k, v in extra.items():
            out[k] = v
    return out


def lincom(fit, weights: dict[str, float], name: str, model_name: str, extra: Optional[dict] = None) -> dict:
    params = fit.params
    cov = fit.cov_params()
    beta = 0.0
    for term, w in weights.items():
        if term not in params.index:
            raise KeyError(f"Term {term!r} not in model parameters")
        beta += w * params[term]
    var = 0.0
    for t1, w1 in weights.items():
        for t2, w2 in weights.items():
            var += w1 * w2 * cov.loc[t1, t2]
    se = math.sqrt(max(var, 0.0))
    z = beta / se if se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    row = {
        "model": model_name,
        "contrast": name,
        "estimate": float(beta),
        "se": float(se),
        "z": float(z),
        "p": float(p),
        "n": int(getattr(fit, "nobs", np.nan)),
    }
    if extra:
        row.update(extra)
    return row


# -----------------------------
# Data preparation
# -----------------------------


@dataclass
class Cols:
    subject: str
    condition: str
    is_up: str
    is_down: str
    rating_raw: str
    rating_z: str
    temp_z: str
    expected_raw: str
    expected_z: str
    success_raw: str
    success_z: str
    trial_in_run_c: Optional[str]
    run_centered_subject: Optional[str]
    global_trial_z_subject: Optional[str]
    up_first: Optional[str]


def infer_state_columns(df: pd.DataFrame) -> Cols:
    return Cols(
        subject=choose_col(df, ["subject", "participant", "sub"], "subject"),
        condition=choose_col(df, ["condition", "reg_condition"], "condition"),
        is_up=choose_col(df, ["is_up", "condition_up"], "is_up"),
        is_down=choose_col(df, ["is_down", "condition_down"], "is_down"),
        rating_raw=choose_col(df, ["rating_raw"], "rating_raw"),
        rating_z=choose_col(df, ["rating_z_subject", "rating_z"], "rating_z"),
        temp_z=choose_col(df, ["temperature_z_subject", "temperature_z"], "temperature_z"),
        expected_raw=choose_col(df, ["expected_pain_raw_passive_model", "expected_pain_raw"], "expected_pain_raw"),
        expected_z=choose_col(df, ["expected_pain_z_passive_model", "expected_pain_z"], "expected_pain_z"),
        success_raw=choose_col(df, ["reg_success_raw_model", "reg_success_raw"], "reg_success_raw"),
        success_z=choose_col(df, ["reg_success_z_model", "reg_success_z", "success"], "reg_success_z"),
        trial_in_run_c=choose_col(df, ["trial_in_run_c"], "trial_in_run_c", required=False),
        run_centered_subject=choose_col(df, ["run_centered_subject", "run_centered"], "run_centered_subject", required=False),
        global_trial_z_subject=choose_col(df, ["global_trial_z_subject"], "global_trial_z_subject", required=False),
        up_first=choose_col(df, ["up_first"], "up_first", required=False),
    )


def prepare_state_table(df: pd.DataFrame) -> tuple[pd.DataFrame, Cols]:
    cols = infer_state_columns(df)
    d = df.copy()
    # Restrict a full trial table to regulation trials.
    if "is_reg" in d.columns:
        d = d.loc[pd.to_numeric(d["is_reg"], errors="coerce").fillna(0).astype(int) == 1].copy()
    elif "is_reg_trial" in d.columns:
        d = d.loc[pd.to_numeric(d["is_reg_trial"], errors="coerce").fillna(0).astype(int) == 1].copy()
    # Standard names used in formulas
    d["subject"] = d[cols.subject].astype(str)
    d["is_up_bin"] = pd.to_numeric(d[cols.is_up], errors="coerce").fillna(0).astype(int)
    d["is_down_bin"] = pd.to_numeric(d[cols.is_down], errors="coerce").fillna(0).astype(int)
    d["rating_raw_y"] = pd.to_numeric(d[cols.rating_raw], errors="coerce")
    d["rating_z_y"] = pd.to_numeric(d[cols.rating_z], errors="coerce")
    d["temp_z"] = pd.to_numeric(d[cols.temp_z], errors="coerce")
    d["expected_raw"] = pd.to_numeric(d[cols.expected_raw], errors="coerce")
    d["expected_z"] = pd.to_numeric(d[cols.expected_z], errors="coerce")
    d["success_raw"] = pd.to_numeric(d[cols.success_raw], errors="coerce")
    d["success_z"] = pd.to_numeric(d[cols.success_z], errors="coerce")
    d["trial_c"] = pd.to_numeric(d[cols.trial_in_run_c], errors="coerce") if cols.trial_in_run_c else 0.0
    d["run_c"] = pd.to_numeric(d[cols.run_centered_subject], errors="coerce") if cols.run_centered_subject else 0.0
    d["global_trial_z"] = pd.to_numeric(d[cols.global_trial_z_subject], errors="coerce") if cols.global_trial_z_subject else 0.0
    d["up_first"] = pd.to_numeric(d[cols.up_first], errors="coerce").fillna(0).astype(int) if cols.up_first else 0

    for col in ["expected_z", "expected_raw", "temp_z", "trial_c", "run_c", "global_trial_z", "up_first"]:
        center(d, col, f"{col}_c")

    # Raw-scale bounded variables. Ratings are 0..200 in ds000140.
    eps = 0.5
    d["rating01"] = (d["rating_raw_y"].clip(0, 200) + eps) / (200 + 2 * eps)
    d["rating01"] = d["rating01"].clip(1e-4, 1 - 1e-4)
    d["rating_logit"] = np.log(d["rating01"] / (1.0 - d["rating01"]))
    d["expected01"] = (d["expected_raw"].clip(0, 200) + eps) / (200 + 2 * eps)
    d["expected01_c"] = d["expected01"] - d["expected01"].mean(skipna=True)

    # Available room in the instructed direction: up = distance to ceiling; down = distance to floor.
    d["available_room_raw"] = np.where(d["is_up_bin"] == 1, 200.0 - d["expected_raw"], d["expected_raw"] - 0.0)
    d.loc[d["available_room_raw"] <= 1e-6, "available_room_raw"] = np.nan
    d["success_fraction_available"] = d["success_raw"] / d["available_room_raw"]
    d["success_fraction_available_clipped"] = d["success_fraction_available"].clip(-1, 1)

    # Margins for descriptive checks.
    d["ceiling_margin_raw"] = 200.0 - d["expected_raw"]
    d["floor_margin_raw"] = d["expected_raw"]

    return d, cols


# -----------------------------
# Analyses
# -----------------------------


def run_core_sensitivity_models(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    contrasts = []

    base_cov = "temp_z_c + trial_c_c + run_c_c + global_trial_z_c + up_first_c + C(subject)"

    model_specs = [
        (
            "success_z_primary_scale",
            "success_z ~ is_up_bin * expected_z_c + " + base_cov,
            "ols",
            ["is_up_bin", "expected_z_c", "is_up_bin:expected_z_c"],
            "Primary success model in z units. Tests original state-dependence claim."
        ),
        (
            "success_raw_raw_scale",
            "success_raw ~ is_up_bin * expected_raw_c + " + base_cov,
            "ols",
            ["is_up_bin", "expected_raw_c", "is_up_bin:expected_raw_c"],
            "Same model in raw rating units. Checks whether effect is not an artefact of z scaling."
        ),
        (
            "observed_rating_z_model",
            "rating_z_y ~ is_up_bin * expected_z_c + " + base_cov,
            "ols",
            ["is_up_bin", "expected_z_c", "is_up_bin:expected_z_c"],
            "Observed regulation rating as outcome. Reduces concern that result is only success-score algebra."
        ),
        (
            "observed_rating_raw_model",
            "rating_raw_y ~ is_up_bin * expected_raw_c + " + base_cov,
            "ols",
            ["is_up_bin", "expected_raw_c", "is_up_bin:expected_raw_c"],
            "Observed raw regulation rating as outcome. Raw-scale companion to observed-rating model."
        ),
        (
            "bounded_logit_transformed_observed_rating",
            "rating_logit ~ is_up_bin * expected01_c + " + base_cov,
            "ols",
            ["is_up_bin", "expected01_c", "is_up_bin:expected01_c"],
            "Bounded-scale sensitivity using logit-transformed raw rating scaled to (0,1). Prefer this if fractional logit is numerically unstable."
        ),
        (
            "fractional_logit_observed_rating_raw_0_200",
            "rating01 ~ is_up_bin * expected01_c + " + base_cov,
            "fraclogit",
            ["is_up_bin", "expected01_c", "is_up_bin:expected01_c"],
            "Bounded-outcome sensitivity using fractional logit on raw rating scaled to (0,1). Check for numerical instability/separation."
        ),
        (
            "available_room_normalized_success",
            "success_fraction_available ~ is_up_bin * expected_raw_c + " + base_cov,
            "ols",
            ["is_up_bin", "expected_raw_c", "is_up_bin:expected_raw_c"],
            "Success divided by available room in the instructed direction. Controls ceiling/floor room."
        ),
        (
            "available_room_normalized_success_clipped",
            "success_fraction_available_clipped ~ is_up_bin * expected_raw_c + " + base_cov,
            "ols",
            ["is_up_bin", "expected_raw_c", "is_up_bin:expected_raw_c"],
            "Clipped available-room sensitivity, limiting extreme fractions to [-1,1]."
        ),
    ]

    for name, formula, kind, terms, note in model_specs:
        try:
            if kind == "fraclogit":
                fit = fit_fractional_logit(d, formula)
            else:
                fit = fit_cluster_ols(d, formula)
            res = extract_terms(fit, name, terms, {"note": note, "formula": formula})
            results.append(res)

            # Derived slopes when model uses is_up_bin * expected_... coding.
            expected_term = [t for t in terms if t.startswith("expected")][0]
            interaction_term = f"is_up_bin:{expected_term}"
            if interaction_term in fit.params.index:
                contrasts.append(pd.DataFrame([
                    lincom(fit, {expected_term: 1.0}, "down_reference_slope", name, {"note": note}),
                    lincom(fit, {expected_term: 1.0, interaction_term: 1.0}, "up_slope", name, {"note": note}),
                    lincom(fit, {interaction_term: 1.0}, "up_minus_down_slope", name, {"note": note}),
                ]))
        except Exception as e:
            results.append(pd.DataFrame([{
                "model": name, "term": "MODEL_FAILED", "beta": np.nan, "se": np.nan,
                "t_or_z": np.nan, "p": np.nan, "n": np.nan, "r2": np.nan,
                "note": note, "formula": formula, "error": str(e)
            }]))

    res_all = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if not res_all.empty:
        res_all["q_fdr_within_all_terms"] = bh_fdr(res_all["p"])

    con_all = pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame()
    if not con_all.empty:
        con_all["q_fdr_within_all_contrasts"] = bh_fdr(con_all["p"])

    return res_all, con_all


def run_bin_models(d: pd.DataFrame, bin_col: str, bin_source: str) -> pd.DataFrame:
    rows = []
    base_cov = "temp_z_c + trial_c_c + run_c_c + up_first_c + C(subject)"
    for b in d[bin_col].dropna().unique():
        sub = d.loc[d[bin_col] == b].copy()
        n = len(sub)
        n_subjects = sub["subject"].nunique()
        if n < 80 or n_subjects < 20:
            rows.append({
                "bin_source": bin_source, "bin": str(b), "model": "success_z_bin_model",
                "term": "SKIPPED_LOW_N", "beta": np.nan, "se": np.nan, "t_or_z": np.nan,
                "p": np.nan, "n": n, "n_subjects": n_subjects, "note": "Skipped: too few trials or subjects"
            })
            continue
        try:
            # Do not include binning variable as covariate if bin source is temperature.
            cov = base_cov if bin_source != "temperature_z_tertile" else "trial_c_c + run_c_c + up_first_c + C(subject)"
            formula = "success_z ~ is_up_bin * expected_z_c + " + cov
            fit = fit_cluster_ols(sub, formula)
            for term in ["is_up_bin", "expected_z_c", "is_up_bin:expected_z_c"]:
                if term in fit.params.index:
                    rows.append({
                        "bin_source": bin_source, "bin": str(b), "model": "success_z_bin_model",
                        "term": term, "beta": float(fit.params[term]), "se": float(fit.bse[term]),
                        "t_or_z": float(fit.tvalues[term]), "p": float(fit.pvalues[term]),
                        "n": int(fit.nobs), "n_subjects": n_subjects, "r2": float(fit.rsquared),
                        "formula": formula,
                    })
        except Exception as e:
            rows.append({
                "bin_source": bin_source, "bin": str(b), "model": "success_z_bin_model",
                "term": "MODEL_FAILED", "beta": np.nan, "se": np.nan, "t_or_z": np.nan,
                "p": np.nan, "n": n, "n_subjects": n_subjects, "error": str(e)
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = bh_fdr(out["p"])
    return out


# -----------------------------
# Cross-fitted passive baseline
# -----------------------------


def infer_behavior_columns(df: pd.DataFrame) -> dict[str, str]:
    return {
        "subject": choose_col(df, ["subject", "participant", "sub"], "subject"),
        "condition": choose_col(df, ["condition", "reg_condition"], "condition"),
        "is_reg": choose_col(df, ["is_reg", "is_reg_trial"], "is_reg", required=False),
        "is_up": choose_col(df, ["is_up", "condition_up"], "is_up", required=False),
        "is_down": choose_col(df, ["is_down", "condition_down"], "is_down", required=False),
        "rating_raw": choose_col(df, ["rating_raw"], "rating_raw"),
        "rating_z": choose_col(df, ["rating_z_subject", "rating_z"], "rating_z"),
        "temp_z": choose_col(df, ["temperature_z_subject", "temperature_z"], "temperature_z"),
        "trial_c": choose_col(df, ["trial_in_run_c"], "trial_in_run_c", required=False),
        "run_c": choose_col(df, ["run_centered_subject", "run_centered"], "run_centered_subject", required=False),
        "up_first": choose_col(df, ["up_first"], "up_first", required=False),
    }


def prepare_behavior_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    c = infer_behavior_columns(df)
    d = df.copy()
    d["subject"] = d[c["subject"]].astype(str)
    cond = d[c["condition"]].astype(str).str.lower()
    d["is_passive"] = cond.eq("passive") | cond.str.contains("passive", na=False)

    if c["is_reg"] and c["is_reg"] in d.columns:
        d["is_reg_bin"] = pd.to_numeric(d[c["is_reg"]], errors="coerce").fillna(0).astype(int)
    else:
        d["is_reg_bin"] = (~d["is_passive"]).astype(int)

    if c["is_up"] and c["is_up"] in d.columns:
        d["is_up_bin"] = pd.to_numeric(d[c["is_up"]], errors="coerce").fillna(0).astype(int)
    else:
        d["is_up_bin"] = cond.str.contains("up", na=False).astype(int)
    if c["is_down"] and c["is_down"] in d.columns:
        d["is_down_bin"] = pd.to_numeric(d[c["is_down"]], errors="coerce").fillna(0).astype(int)
    else:
        d["is_down_bin"] = cond.str.contains("down", na=False).astype(int)

    d["rating_raw_y"] = pd.to_numeric(d[c["rating_raw"]], errors="coerce")
    d["rating_z_y"] = pd.to_numeric(d[c["rating_z"]], errors="coerce")
    d["temp_z"] = pd.to_numeric(d[c["temp_z"]], errors="coerce")
    d["trial_c"] = pd.to_numeric(d[c["trial_c"]], errors="coerce") if c["trial_c"] else 0.0
    d["run_c"] = pd.to_numeric(d[c["run_c"]], errors="coerce") if c["run_c"] else 0.0
    d["up_first"] = pd.to_numeric(d[c["up_first"]], errors="coerce").fillna(0).astype(int) if c["up_first"] else 0
    return d, c


def fit_subject_passive_model(train: pd.DataFrame, y: str):
    # Fit the prespecified within-subject passive model first, followed by reduced alternatives if required.
    formulas = [
        f"{y} ~ temp_z + trial_c + run_c",
        f"{y} ~ temp_z + trial_c",
        f"{y} ~ temp_z",
    ]
    last_err = None
    for f in formulas:
        try:
            fit = smf.ols(f, data=train).fit()
            if np.all(np.isfinite(fit.params.values)):
                return fit, f
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not fit passive model for {y}: {last_err}")


def crossfit_passive_baseline(beh: pd.DataFrame, k: int = 5, random_seed: int = 140) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    outputs = []
    logs = []
    for subj, sdf in beh.groupby("subject", sort=True):
        sdf = sdf.copy()
        passive_idx = sdf.index[sdf["is_passive"]].to_numpy()
        reg = sdf.loc[sdf["is_reg_bin"] == 1].copy()
        if len(passive_idx) < max(k * 3, 15) or reg.empty:
            logs.append({"subject": subj, "status": "SKIPPED_LOW_PASSIVE_OR_NO_REG", "n_passive": len(passive_idx), "n_reg": len(reg)})
            continue
        # Deterministic fold assignment within subject.
        shuffled = passive_idx.copy()
        rng.shuffle(shuffled)
        fold_map = {idx: i % k for i, idx in enumerate(shuffled)}
        preds = {"z": [], "raw": []}
        fold_logs = []
        for fold in range(k):
            train_idx = [idx for idx in passive_idx if fold_map[idx] != fold]
            train = sdf.loc[train_idx].copy()
            if len(train) < 10:
                continue
            try:
                fit_z, formula_z = fit_subject_passive_model(train, "rating_z_y")
                fit_raw, formula_raw = fit_subject_passive_model(train, "rating_raw_y")
                pred_z = fit_z.predict(reg)
                pred_raw = fit_raw.predict(reg)
                preds["z"].append(np.asarray(pred_z, dtype=float))
                preds["raw"].append(np.asarray(pred_raw, dtype=float))
                fold_logs.append({"fold": fold, "formula_z": formula_z, "formula_raw": formula_raw, "n_train": len(train)})
            except Exception as e:
                fold_logs.append({"fold": fold, "error": str(e), "n_train": len(train)})
        if not preds["z"] or not preds["raw"]:
            logs.append({"subject": subj, "status": "FAILED_ALL_FOLDS", "fold_logs": json.dumps(fold_logs), "n_passive": len(passive_idx), "n_reg": len(reg)})
            continue
        reg["expected_z_crossfit"] = np.vstack(preds["z"]).mean(axis=0)
        reg["expected_raw_crossfit"] = np.vstack(preds["raw"]).mean(axis=0)
        reg["success_z_crossfit"] = np.where(reg["is_up_bin"] == 1, reg["rating_z_y"] - reg["expected_z_crossfit"], reg["expected_z_crossfit"] - reg["rating_z_y"])
        reg["success_raw_crossfit"] = np.where(reg["is_up_bin"] == 1, reg["rating_raw_y"] - reg["expected_raw_crossfit"], reg["expected_raw_crossfit"] - reg["rating_raw_y"])
        outputs.append(reg)
        logs.append({"subject": subj, "status": "OK", "n_passive": len(passive_idx), "n_reg": len(reg), "fold_logs": json.dumps(fold_logs)})
    if not outputs:
        return pd.DataFrame(), pd.DataFrame(logs)
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(logs)


def run_crossfit_models(cf: pd.DataFrame) -> pd.DataFrame:
    if cf.empty:
        return pd.DataFrame()
    d = cf.copy()
    # Standard covariates and centered predictors.
    d["expected_z_crossfit_c"] = d["expected_z_crossfit"] - d["expected_z_crossfit"].mean(skipna=True)
    d["expected_raw_crossfit_c"] = d["expected_raw_crossfit"] - d["expected_raw_crossfit"].mean(skipna=True)
    for c in ["temp_z", "trial_c", "run_c", "up_first"]:
        d[f"{c}_c"] = d[c] - d[c].mean(skipna=True)
    eps = 0.5
    d["rating01"] = (d["rating_raw_y"].clip(0, 200) + eps) / (200 + 2 * eps)
    d["rating01"] = d["rating01"].clip(1e-4, 1 - 1e-4)
    d["rating_logit"] = np.log(d["rating01"] / (1.0 - d["rating01"]))
    d["expected01_crossfit"] = (d["expected_raw_crossfit"].clip(0, 200) + eps) / (200 + 2 * eps)
    d["expected01_crossfit_c"] = d["expected01_crossfit"] - d["expected01_crossfit"].mean(skipna=True)
    d["available_room_raw_crossfit"] = np.where(d["is_up_bin"] == 1, 200 - d["expected_raw_crossfit"], d["expected_raw_crossfit"])
    d.loc[d["available_room_raw_crossfit"] <= 1e-6, "available_room_raw_crossfit"] = np.nan
    d["success_fraction_crossfit"] = d["success_raw_crossfit"] / d["available_room_raw_crossfit"]

    specs = [
        ("crossfit_success_z", "success_z_crossfit ~ is_up_bin * expected_z_crossfit_c + temp_z_c + trial_c_c + run_c_c + up_first_c + C(subject)", "ols", ["is_up_bin", "expected_z_crossfit_c", "is_up_bin:expected_z_crossfit_c"]),
        ("crossfit_success_raw", "success_raw_crossfit ~ is_up_bin * expected_raw_crossfit_c + temp_z_c + trial_c_c + run_c_c + up_first_c + C(subject)", "ols", ["is_up_bin", "expected_raw_crossfit_c", "is_up_bin:expected_raw_crossfit_c"]),
        ("crossfit_observed_raw", "rating_raw_y ~ is_up_bin * expected_raw_crossfit_c + temp_z_c + trial_c_c + run_c_c + up_first_c + C(subject)", "ols", ["is_up_bin", "expected_raw_crossfit_c", "is_up_bin:expected_raw_crossfit_c"]),
        ("crossfit_bounded_logit_observed", "rating_logit ~ is_up_bin * expected01_crossfit_c + temp_z_c + trial_c_c + run_c_c + up_first_c + C(subject)", "ols", ["is_up_bin", "expected01_crossfit_c", "is_up_bin:expected01_crossfit_c"]),
        ("crossfit_fractional_logit_observed", "rating01 ~ is_up_bin * expected01_crossfit_c + temp_z_c + trial_c_c + run_c_c + up_first_c + C(subject)", "fraclogit", ["is_up_bin", "expected01_crossfit_c", "is_up_bin:expected01_crossfit_c"]),
        ("crossfit_available_room_normalized", "success_fraction_crossfit ~ is_up_bin * expected_raw_crossfit_c + temp_z_c + trial_c_c + run_c_c + up_first_c + C(subject)", "ols", ["is_up_bin", "expected_raw_crossfit_c", "is_up_bin:expected_raw_crossfit_c"]),
    ]
    rows = []
    con = []
    for name, formula, kind, terms in specs:
        try:
            fit = fit_fractional_logit(d, formula) if kind == "fraclogit" else fit_cluster_ols(d, formula)
            rows.append(extract_terms(fit, name, terms, {"formula": formula}))
            exp_terms = [t for t in terms if t.startswith("expected")]
            if exp_terms:
                exp = exp_terms[0]
                inter = f"is_up_bin:{exp}"
                if inter in fit.params.index:
                    con.extend([
                        lincom(fit, {exp: 1}, "down_reference_slope", name),
                        lincom(fit, {exp: 1, inter: 1}, "up_slope", name),
                        lincom(fit, {inter: 1}, "up_minus_down_slope", name),
                    ])
        except Exception as e:
            rows.append(pd.DataFrame([{"model": name, "term": "MODEL_FAILED", "p": np.nan, "error": str(e), "formula": formula}]))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out["q_fdr_terms"] = bh_fdr(out["p"])
    con_df = pd.DataFrame(con)
    if not con_df.empty:
        con_df["term"] = con_df["contrast"]
        con_df["beta"] = con_df["estimate"]
        con_df["t_or_z"] = con_df["z"]
        con_df["q_fdr_terms"] = bh_fdr(con_df["p"])
        out = pd.concat([out, con_df[out.columns.intersection(con_df.columns).tolist() + [c for c in con_df.columns if c not in out.columns]]], ignore_index=True, sort=False)
    return out


def descriptive_margins(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, sub in [("all", d), ("up", d[d["is_up_bin"] == 1]), ("down", d[d["is_down_bin"] == 1])]:
        for col in ["expected_raw", "rating_raw_y", "success_raw", "ceiling_margin_raw", "floor_margin_raw", "available_room_raw", "success_fraction_available"]:
            s = pd.to_numeric(sub[col], errors="coerce")
            rows.append({
                "group": label, "variable": col, "n": int(s.notna().sum()),
                "mean": float(s.mean()), "sd": float(s.std(ddof=1)), "min": float(s.min()),
                "p05": float(s.quantile(0.05)), "median": float(s.median()), "p95": float(s.quantile(0.95)), "max": float(s.max())
            })
    # near-bound counts
    for label, sub in [("all", d), ("up", d[d["is_up_bin"] == 1]), ("down", d[d["is_down_bin"] == 1])]:
        rows.append({"group": label, "variable": "n_expected_within_10_raw_units_of_ceiling", "n": int((sub["ceiling_margin_raw"] <= 10).sum())})
        rows.append({"group": label, "variable": "n_expected_within_10_raw_units_of_floor", "n": int((sub["floor_margin_raw"] <= 10).sum())})
        rows.append({"group": label, "variable": "n_rating_at_or_above_190", "n": int((sub["rating_raw_y"] >= 190).sum())})
        rows.append({"group": label, "variable": "n_rating_at_or_below_10", "n": int((sub["rating_raw_y"] <= 10).sum())})
    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------


def main():
    ap = argparse.ArgumentParser(description="Run state-dependence sensitivity controls for ds000140.")
    ap.add_argument("--project", type=Path, default=Path("."))
    ap.add_argument("--state-table", type=Path, default=None)
    ap.add_argument("--behavior-table", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--crossfit-k", type=int, default=5)
    ap.add_argument("--random-seed", type=int, default=140)
    args = ap.parse_args()

    project = args.project
    outdir = args.outdir or project / "results" / "sensitivity_audits" / "state_dependence_sensitivity_controls"
    outdir.mkdir(parents=True, exist_ok=True)

    state_path = args.state_table or first_existing([
        project / "results/behavior/trialwise_behavior_latents.tsv",
    ])
    if state_path is None:
        raise FileNotFoundError("Could not find regulation state table. Provide --state-table explicitly.")

    behavior_path = args.behavior_table or first_existing([
        project / "results/behavior/trialwise_behavior_latents.tsv",
        project / "results/behavior/behavior_trials_master_clean.tsv.gz",
    ])

    state_raw = pd.read_csv(state_path, sep="\t")
    state, cols = prepare_state_table(state_raw)

    # Core model sensitivities.
    terms, contrasts = run_core_sensitivity_models(state)
    terms.to_csv(outdir / "sensitivity_model_terms.tsv", sep="\t", index=False)
    contrasts.to_csv(outdir / "sensitivity_model_contrasts.tsv", sep="\t", index=False)

    # Binning analyses.
    state["temperature_z_tertile"] = safe_qcut(state["temp_z"], 3, ["low_temperature", "mid_temperature", "high_temperature"])
    state["expected_pain_z_tertile"] = safe_qcut(state["expected_z"], 3, ["low_expected", "mid_expected", "high_expected"])
    stim_bins = run_bin_models(state, "temperature_z_tertile", "temperature_z_tertile")
    expected_bins = run_bin_models(state, "expected_pain_z_tertile", "expected_pain_z_tertile")
    stim_bins.to_csv(outdir / "stimulus_bin_results.tsv", sep="\t", index=False)
    expected_bins.to_csv(outdir / "expected_pain_bin_results.tsv", sep="\t", index=False)

    desc = descriptive_margins(state)
    desc.to_csv(outdir / "descriptives_and_margins.tsv", sep="\t", index=False)

    # Cross-fitted passive baseline if a full behaviour table is present.
    crossfit_status = "SKIPPED_NO_BEHAVIOR_TABLE"
    if behavior_path is not None and behavior_path.exists():
        beh_raw = pd.read_csv(behavior_path, sep="\t")
        beh, bc = prepare_behavior_table(beh_raw)
        cf, cf_log = crossfit_passive_baseline(beh, k=args.crossfit_k, random_seed=args.random_seed)
        cf.to_csv(outdir / "crossfit_passive_baseline_trial_table.tsv", sep="\t", index=False)
        cf_log.to_csv(outdir / "crossfit_passive_baseline_log.tsv", sep="\t", index=False)
        cf_results = run_crossfit_models(cf)
        cf_results.to_csv(outdir / "crossfit_passive_baseline_results.tsv", sep="\t", index=False)
        crossfit_status = "OK" if not cf.empty else "FAILED_OR_EMPTY"
    else:
        pd.DataFrame([{"status": "SKIPPED", "reason": "No full behaviour table found or provided"}]).to_csv(outdir / "crossfit_passive_baseline_results.tsv", sep="\t", index=False)

    # Create a concise table of target interaction and contrast rows.
    summary_parts = []
    if not contrasts.empty:
        c = contrasts.copy()
        c = c.loc[c["contrast"].isin(["up_minus_down_slope", "up_slope", "down_reference_slope"])].copy()
        c["source_table"] = "sensitivity_model_contrasts.tsv"
        summary_parts.append(c)
    cf_path = outdir / "crossfit_passive_baseline_results.tsv"
    if cf_path.exists():
        cf_res = pd.read_csv(cf_path, sep="\t")
        if "term" in cf_res.columns:
            cf_target = cf_res.loc[cf_res["term"].astype(str).str.contains("up_minus_down_slope|up_slope|down_reference_slope|is_up_bin:expected", regex=True, na=False)].copy()
            cf_target["source_table"] = "crossfit_passive_baseline_results.tsv"
            summary_parts.append(cf_target)
    if summary_parts:
        summary = pd.concat(summary_parts, ignore_index=True, sort=False)
        if "p" in summary.columns:
            summary["q_fdr_summary"] = bh_fdr(summary["p"])
        summary.to_csv(outdir / "state_dependence_sensitivity_summary.tsv", sep="\t", index=False)

    metadata = {
        "project": str(project),
        "state_table": str(state_path),
        "behavior_table": str(behavior_path) if behavior_path else None,
        "outdir": str(outdir),
        "n_regulation_trials": int(len(state)),
        "n_subjects": int(state["subject"].nunique()),
        "crossfit_status": crossfit_status,
        "crossfit_k": args.crossfit_k,
        "random_seed": args.random_seed,
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme = f"""# State-dependence sensitivity controls

## Inputs
- Regulation state table: `{state_path}`
- Full behaviour table: `{behavior_path if behavior_path else 'not available'}`

## What this script tests
1. Original state-dependence model in standardized success units.
2. Same state-dependence model in raw rating units.
3. Observed-rating models, so the result is not only presented as success-score algebra.
4. Bounded-scale models for raw ratings scaled to (0,1): logit-transformed OLS and fractional-logit GLM.
5. Available-room-normalized success models to control ceiling/floor room in the instructed direction.
6. Stimulus-intensity tertile analyses.
7. Expected-pain tertile analyses.
8. Cross-fitted passive-baseline sensitivity models, if the full behavioural table is available.

## Key files to inspect first
- `state_dependence_sensitivity_summary.tsv`
- `sensitivity_model_contrasts.tsv`
- `crossfit_passive_baseline_results.tsv`
- `stimulus_bin_results.tsv`
- `expected_pain_bin_results.tsv`
- `descriptives_and_margins.tsv`

## Interpretation guide
- The principal term/contrast is `up_minus_down_slope` or the interaction term containing `is_up_bin:expected...`.
- Persistence across raw-scale, observed-rating, bounded-outcome, available-room-normalized and cross-fitted-baseline models would support robustness to bounded-scale and passive-baseline coupling.
- Attenuation after available-room normalization or bounded-outcome modelling indicates sensitivity to outcome scaling rather than a stable state-dependent mechanism.
- Stimulus-bin results are descriptive robustness analyses and may be unstable in small bins.
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")

    print("State-dependence sensitivity controls completed.")
    print(f"Output directory: {outdir}")
    print(f"State table: {state_path}")
    print(f"Behaviour table: {behavior_path if behavior_path else 'not available'}")
    print(f"Crossfit status: {crossfit_status}")


if __name__ == "__main__":
    main()
