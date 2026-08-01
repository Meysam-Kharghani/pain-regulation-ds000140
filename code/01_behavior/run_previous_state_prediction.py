#!/usr/bin/env python3
"""
run_previous_state_prediction.py

Strict previous-only prediction of pain-regulation success.

Leakage rule:
For prediction of trial t, do not use any information from trial t itself.
Because from the onset of trial t the subject has already started regulation.

Therefore, by default this script excludes:
- current condition/is_up/is_down as predictors
- current temperature
- current expected_pain
- current rating
- current success/outcome derivatives
- current same-trial neural LSS/NPS/ROI values

Allowed predictors:
- previous-trial behavior and previous regulation success
- cumulative history computed before trial t
- previous-trial neural features
- optional pre-onset ROI state extracted strictly before onset_t - safety_gap
- subject-level trait/domain features, designated as individual-difference features
- time/order variables that are known before the trial, but reported separately

Models are evaluated:
- all regulation trials, condition-blind
- up trials only, condition-stratified evaluation but condition is not a predictor
- down trials only, condition-stratified evaluation but condition is not a predictor
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


STRICT_EXCLUDE_CURRENT_PATTERNS = [
    "current_z", "_current", "rating_for_model", "rating_z_subject", "rating_raw",
    "success", "reg_success", "expected_pain", "temperature_z_subject", "temperature",
    "condition", "is_up", "is_down", "is_reg", "is_passive",
    "observed", "response", "outcome", "stimulus", "stim_", "heat",
]

ALLOWED_CURRENT_PREONSET_SUFFIX = "_preonset_zmean"


def mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_read_tsv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="	")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return x * np.nan
    return (x - x.mean()) / sd


def fdr_bh(pvals: List[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q_ok = np.empty_like(q_ranked)
    q_ok[order] = q_ranked
    q[ok] = q_ok
    return q


def sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    if re.match(r"^[0-9]", s):
        s = "f_" + s
    return s[:120]


def infer_trialwise_path(derivatives: Path) -> Optional[Path]:
    candidates = [
        derivatives / "computational_pain_regulation_analysis" / "tables" / "trialwise_model_predictions.tsv",
        derivatives / "trialwise_model_predictions.tsv",
        derivatives / "trialwise_state_success_prediction_analysis" / "tables" / "trialwise_state_table.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    found = list(derivatives.glob("**/trialwise_model_predictions.tsv"))
    return found[0] if found else None


def infer_prev_neural_table(derivatives: Path) -> Optional[Path]:
    candidates = [
        derivatives / "trialwise_neural_state_integration_analysis" / "tables" / "trialwise_neural_state_table.tsv",
        derivatives / "trialwise_neural_state_integration_analysis" / "tables" / "trialwise_neural_state_table.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    found = list(derivatives.glob("**/trialwise_neural_state_table.tsv"))
    return found[0] if found else None


def infer_preonset_table(derivatives: Path) -> Optional[Path]:
    candidates = [
        derivatives / "preonset_roi_state_analysis" / "tables" / "preonset_roi_state.tsv",
        derivatives / "preonset_roi_state_analysis" / "tables" / "preonset_roi_state.tsv",
        derivatives / "preonset_roi_state_analysis" / "tables" / "preonset_roi_state.tsv",
        derivatives / "preonset_roi_state_analysis" / "tables" / "preonset_roi_state.tsv",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 5:
            return p
    found = list(derivatives.glob("**/preonset_roi_state.tsv"))
    for p in found:
        try:
            if p.exists() and p.stat().st_size > 5:
                return p
        except Exception:
            pass
    return None


def infer_domain_scores(derivatives: Path) -> Optional[Path]:
    candidates = [
        derivatives / "refined_individual_fingerprint_analysis" / "tables" / "domain_scores.tsv",
        derivatives / "domain_scores.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["sub", "participant_id", "subject_id"]:
        if c in out.columns and "subject" not in out.columns:
            out = out.rename(columns={c: "subject"})
    if "subject" in out.columns:
        out["subject"] = out["subject"].astype(str).str.replace("^sub-", "", regex=True)
        out["subject"] = out["subject"].apply(lambda x: x if str(x).startswith("sub-") else f"sub-{x}")
    if "run_label" in out.columns and "run" not in out.columns:
        out["run"] = pd.to_numeric(out["run_label"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    for c in ["run_id", "run_index", "run_number"]:
        if c in out.columns and "run" not in out.columns:
            out["run"] = pd.to_numeric(out[c], errors="coerce")
    if "run" in out.columns:
        out["run"] = pd.to_numeric(out["run"], errors="coerce")

    if "trial_in_run" not in out.columns:
        for c in ["trial_lss_index", "trial_index", "trial_num", "trial", "trial_number", "event_index"]:
            if c in out.columns:
                out = out.rename(columns={c: "trial_in_run"})
                break
    if "trial_in_run" in out.columns:
        out["trial_in_run"] = pd.to_numeric(out["trial_in_run"], errors="coerce")

    if "condition" in out.columns:
        out["condition"] = out["condition"].astype(str).str.lower()
    return out


def ensure_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "condition" not in out.columns:
        if "is_up" in out.columns and "is_down" in out.columns:
            out["condition"] = np.where(out["is_up"].astype(float).eq(1), "up",
                                        np.where(out["is_down"].astype(float).eq(1), "down", "passive"))
        else:
            out["condition"] = "unknown"
    if "is_up" not in out.columns:
        out["is_up"] = out["condition"].astype(str).str.lower().eq("up").astype(int)
    if "is_down" not in out.columns:
        out["is_down"] = out["condition"].astype(str).str.lower().eq("down").astype(int)
    if "is_reg" not in out.columns:
        out["is_reg"] = ((out["is_up"].astype(float) == 1) | (out["is_down"].astype(float) == 1)).astype(int)
    return out


def pick_success_col(df: pd.DataFrame) -> str:
    for c in ["reg_success_z_model", "reg_success_z", "success", "reg_success_raw_model", "reg_success_raw"]:
        if c in df.columns:
            return c
    raise ValueError("No success column found.")


def make_previous_behavior_features(trialwise: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = ensure_flags(harmonize(trialwise))
    success_col = pick_success_col(df)

    df["success"] = pd.to_numeric(df[success_col], errors="coerce")

    if "global_trial_index" not in df.columns:
        if "run" in df.columns and "trial_in_run" in df.columns:
            df["global_trial_index"] = df["run"].fillna(0) * 1000 + df["trial_in_run"].fillna(0)
        else:
            df["global_trial_index"] = df.groupby("subject").cumcount() + 1

    df = df.sort_values(["subject", "global_trial_index"]).copy()

    # Candidate previous-only behavior sources.
    prev_source_cols = {
        "rating_z_subject": "prev_rating_z",
        "rating_raw": "prev_rating_raw",
        "reg_success_z_model": "prev_success_any",
        "reg_success_z": "prev_success_any_alt",
        "expected_pain_z_passive_model": "prev_expected_pain",
        "expected_pain_z": "prev_expected_pain_alt",
        "temperature_z_subject": "prev_temperature_z",
        "temperature": "prev_temperature",
    }
    for src, dst in prev_source_cols.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")
            df[dst] = df.groupby("subject")[dst].shift(1)

    # Previous regulation success only.
    df["prev_reg_success"] = np.nan
    df["prev_reg_condition_was_up"] = np.nan
    for sub, g in df.groupby("subject", sort=False):
        last_success = np.nan
        last_up = np.nan
        vals_s, vals_u = [], []
        for _, row in g.iterrows():
            vals_s.append(last_success)
            vals_u.append(last_up)
            if int(row.get("is_reg", 0)) == 1 and pd.notna(row.get("success", np.nan)):
                last_success = row.get("success", np.nan)
                last_up = row.get("is_up", np.nan)
        df.loc[g.index, "prev_reg_success"] = vals_s
        df.loc[g.index, "prev_reg_condition_was_up"] = vals_u

    # Cumulative history BEFORE current trial.
    for src, dst in [
        ("rating_z_subject", "cum_prev_rating_mean"),
        ("reg_success_z_model", "cum_prev_success_mean"),
        ("expected_pain_z_passive_model", "cum_prev_expected_mean"),
    ]:
        if src in df.columns:
            vals = pd.to_numeric(df[src], errors="coerce")
            df["_working_values"] = vals
            df[dst] = df.groupby("subject")["_working_values"].transform(lambda s: s.expanding().mean().shift(1))
            df.drop(columns=["_working_values"], inplace=True)

    if "rating_z_subject" in df.columns and "condition" in df.columns:
        df["_passive_working_values"] = np.where(df["condition"].eq("passive"), pd.to_numeric(df["rating_z_subject"], errors="coerce"), np.nan)
        df["cum_prev_passive_rating_mean"] = df.groupby("subject")["_passive_working_values"].transform(lambda s: s.expanding().mean().shift(1))
        df.drop(columns=["_passive_working_values"], inplace=True)

    # Time/order known before trial: kept separately.
    if "global_trial_z_subject" not in df.columns:
        df["global_trial_z_subject"] = df.groupby("subject")["global_trial_index"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else np.nan))
    if "up_first" in df.columns:
        df["up_first"] = pd.to_numeric(df["up_first"], errors="coerce")

    reg = df[df["is_reg"].astype(float) == 1].copy()
    info = {
        "success_col": success_col,
        "n_reg_trials": int(len(reg)),
        "n_subjects": int(reg["subject"].nunique()),
    }
    return reg, info


def merge_by_keys(base: pd.DataFrame, other: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    b = base.copy()
    o = harmonize(other.copy())
    keys = [k for k in ["subject", "run", "trial_in_run"] if k in b.columns and k in o.columns]
    if len(keys) < 3:
        return b
    # Avoid duplicate nonfeature columns.
    keep = keys + [c for c in o.columns if c not in keys and c not in b.columns]
    return b.merge(o[keep].drop_duplicates(keys), on=keys, how="left")


def is_valid_previous_neural_col(c: str) -> bool:
    """
    .1: accept true previous-neural features even if the source prefix
    contains words like 'trialwise' or 'tables'.

    Allowed predictor classes:
    - tables_trialwise_nps_lss_nps_dot_lss_prev_z
    - tables_trialwise_roi_lss_wide_roi_reg_dlPFC_IFJ_L_prev_z

    Excluded predictor classes:
    - rating_for_model_prev_z
    - onset_prev_z
    - design_rank_prev_z
    - expected_pain_prev_z
    """
    lc = c.lower()
    if not lc.endswith("_prev_z"):
        return False

    # Apply strong leakage and design rejection rules. Do not reject a variable solely because the prefix
    # contains 'trialwise'.
    bad_tokens = [
        "rating_for_model", "rating_z", "rating_raw", "success",
        "expected_pain", "temperature", "onset", "duration",
        "design", "rank", "cols", "n_vol", "motion", "fd", "dvars",
        "condition", "is_up", "is_down", "is_reg", "is_passive",
        "global_trial", "trial_in_run_prev", "run_prev"
    ]
    if any(tok in lc for tok in bad_tokens):
        return False

    # Require evidence that it is an actual neural feature.
    good_tokens = [
        "nps_dot", "nps_l2norm", "nps_weighted", "signature",
        "_roi_", "_roi_", "roi_reg_", "roi_sensory_", "roi_expl_",
        "_edge_", "_to_", "fc_", "ec_", "connectivity"
    ]
    return any(tok in lc for tok in good_tokens)


def attach_previous_neural(base: pd.DataFrame, prev_neural_path: Optional[Path]) -> Tuple[pd.DataFrame, List[str]]:
    if prev_neural_path is None or not prev_neural_path.exists():
        return base, []
    nt = safe_read_tsv(prev_neural_path)
    nt = harmonize(nt)

    allowed = [c for c in nt.columns if is_valid_previous_neural_col(c)]

    keys = [k for k in ["subject", "run", "trial_in_run"] if k in base.columns and k in nt.columns]
    if len(keys) < 3 or not allowed:
        return base, []

    merged = base.merge(nt[keys + allowed].drop_duplicates(keys), on=keys, how="left")
    return merged, allowed


def attach_preonset(base: pd.DataFrame, preonset_path: Optional[Path]) -> Tuple[pd.DataFrame, List[str]]:
    if preonset_path is None or not preonset_path.exists() or preonset_path.stat().st_size <= 5:
        return base, []
    pt = safe_read_tsv(preonset_path)
    if pt is None or pt.empty:
        return base, []
    pt = harmonize(pt)
    allowed = [c for c in pt.columns if c.endswith(ALLOWED_CURRENT_PREONSET_SUFFIX)]
    keys = [k for k in ["subject", "run", "trial_in_run"] if k in base.columns and k in pt.columns]
    if len(keys) < 3 or not allowed:
        return base, []
    merged = base.merge(pt[keys + allowed].drop_duplicates(keys), on=keys, how="left")
    return merged, allowed


def attach_domains(base: pd.DataFrame, domain_path: Optional[Path]) -> Tuple[pd.DataFrame, List[str]]:
    if domain_path is None or not domain_path.exists():
        return base, []
    dom = safe_read_tsv(domain_path)
    dom = harmonize(dom)
    if "subject" not in dom.columns:
        return base, []
    cols = []
    for c in dom.columns:
        if c == "subject":
            continue
        vals = pd.to_numeric(dom[c], errors="coerce")
        if vals.notna().sum() >= 5 and vals.nunique(dropna=True) > 1:
            dom[c + "_domain_z"] = zscore(vals)
            cols.append(c + "_domain_z")
    merged = base.merge(dom[["subject"] + cols], on="subject", how="left")
    return merged, cols


def is_strict_allowed_feature(c: str, source: str) -> Tuple[bool, str]:
    lc = c.lower()
    if c in ["subject", "run", "trial_in_run", "condition", "success"]:
        return False, "identifier_or_outcome"
    if source == "preonset":
        if c.endswith(ALLOWED_CURRENT_PREONSET_SUFFIX):
            return True, "preonset_before_current_onset"
        return False, "not_preonset"
    if source in ["previous_behavior", "previous_neural", "domain", "time_order"]:
        # Previous features may contain prev_expected, prev_temperature, prev_success. That is allowed.
        if source == "previous_behavior":
            if c.startswith("prev_") or c.startswith("cum_prev_"):
                return True, "previous_or_cumulative_behavior"
        if source == "previous_neural":
            if c.endswith("_prev_z"):
                return True, "previous_trial_neural"
        if source == "domain":
            if c.endswith("_domain_z"):
                return True, "subject_level_domain"
        if source == "time_order":
            if c in ["global_trial_z_subject", "up_first"]:
                return True, "known_time_or_order"
        return False, "not_in_source_whitelist"
    # default strict blacklist
    if any(p in lc for p in STRICT_EXCLUDE_CURRENT_PATTERNS):
        return False, "current_trial_or_leakage_pattern"
    return True, "passed"


def build_feature_audit(df: pd.DataFrame, prev_neural_cols: List[str], preonset_cols: List[str], domain_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    previous_behavior = [c for c in df.columns if c.startswith("prev_") or c.startswith("cum_prev_")]
    time_order = [c for c in ["global_trial_z_subject", "up_first"] if c in df.columns]

    sources = {
        "previous_behavior": previous_behavior,
        "time_order": time_order,
        "previous_neural": prev_neural_cols,
        "preonset": preonset_cols,
        "domain": domain_cols,
    }

    audit_rows = []
    allowed_by_source = {}
    for src, cols in sources.items():
        allowed = []
        for c in cols:
            ok, reason = is_strict_allowed_feature(c, src)
            vals = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(dtype=float)
            if ok and vals.notna().sum() >= 20 and vals.nunique(dropna=True) > 1:
                allowed.append(c)
                ok = True
                reason = reason
            else:
                ok = False
                reason = reason if ok else reason
                if ok:
                    reason = "too_missing_or_constant"
            audit_rows.append({
                "feature": c,
                "source": src,
                "allowed": int(ok),
                "reason": reason,
                "n_nonmissing": int(vals.notna().sum()) if len(vals) else 0,
                "n_unique": int(vals.nunique(dropna=True)) if len(vals) else 0,
            })
        allowed_by_source[src] = allowed
    return pd.DataFrame(audit_rows), allowed_by_source


def cluster_ols(formula: str, data: pd.DataFrame) -> pd.DataFrame:
    if not HAS_STATSMODELS:
        return pd.DataFrame([{"model_ok": 0, "error": "statsmodels unavailable", "formula": formula}])
    d = data.copy()
    rhs = formula.replace("~", "+").replace("*", "+").replace(":", "+").replace("C(subject)", "subject")
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rhs)
    needed = []
    for t in tokens:
        if t in d.columns and t not in needed:
            needed.append(t)
    if "success" in d.columns and "success" not in needed:
        needed = ["success"] + needed
    if "subject" not in needed and "subject" in d.columns:
        needed.append("subject")
    d_fit = d.dropna(subset=needed).copy()
    try:
        if len(d_fit) < 20:
            raise ValueError(f"too few rows after NA drop: {len(d_fit)}")
        model = smf.ols(formula, data=d_fit).fit(cov_type="cluster", cov_kwds={"groups": d_fit["subject"].values})
        rows = []
        for term in model.params.index:
            rows.append({
                "formula": formula,
                "term": term,
                "beta": model.params.get(term, np.nan),
                "se": model.bse.get(term, np.nan),
                "t_or_z": model.tvalues.get(term, np.nan),
                "p": model.pvalues.get(term, np.nan),
                "n": int(model.nobs),
                "r2": float(getattr(model, "rsquared", np.nan)),
                "model_ok": 1,
            })
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame([{"formula": formula, "term": "_MODEL_FAILED_", "model_ok": 0, "error": str(e), "n": len(d_fit)}])


def screen_correlations(df: pd.DataFrame, cols: List[str], top_n: int = 30) -> pd.DataFrame:
    rows = []
    for c in cols:
        valid_rows = df[[c, "success"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(valid_rows) < 30 or valid_rows[c].nunique() < 3:
            continue
        if HAS_SCIPY:
            r, p = stats.pearsonr(valid_rows[c], valid_rows["success"])
        else:
            r, p = valid_rows[c].corr(valid_rows["success"]), np.nan
        rows.append({"feature": c, "r": r, "abs_r": abs(r), "p": p, "n": len(valid_rows)})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_fdr"] = fdr_bh(out["p"].tolist())
        out = out.sort_values("abs_r", ascending=False).head(top_n)
    return out


def run_mechanism_tests(df: pd.DataFrame, allowed: Dict[str, List[str]], top_n: int = 25) -> pd.DataFrame:
    rows = []
    base = allowed.get("previous_behavior", []) + allowed.get("time_order", [])
    # Keep compact behavior terms.
    base = [c for c in base if c in df.columns][:12]
    if base:
        formula = "success ~ " + " + ".join(base) + " + C(subject)"
        out = cluster_ols(formula, df)
        out.insert(0, "model_name", "strict_previous_behavior_subject_FE")
        out.insert(1, "feature_source", "previous_behavior")
        rows.append(out)

    for src in ["previous_neural", "preonset", "domain"]:
        cols = allowed.get(src, [])
        screen = screen_correlations(df, cols, top_n=top_n)
        for c in screen.get("feature", []):
            if src == "domain":
                # domains are collinear with subject FE; no C(subject)
                formula = "success ~ " + " + ".join(base + [c])
            else:
                formula = "success ~ " + " + ".join(base + [c]) + " + C(subject)"
            out = cluster_ols(formula, df)
            out.insert(0, "model_name", f"strict_{src}_{c}")
            out.insert(1, "feature_source", src)
            rows.append(out)

    res = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(res):
        res["is_target_term"] = (~res["term"].astype(str).str.startswith("C(subject)")) & (res["term"].astype(str) != "Intercept") & res["p"].notna()
        res["q_fdr_target_terms"] = np.nan
        mask = res["is_target_term"]
        res.loc[mask, "q_fdr_target_terms"] = fdr_bh(res.loc[mask, "p"].tolist())
    return res


def cv_splits_forward_subject(df: pd.DataFrame, test_frac: float = 0.3):
    train_idx = []
    test_idx = []
    for sub, g in df.sort_values(["subject", "global_trial_index"]).groupby("subject"):
        idx = list(g.index)
        if len(idx) < 4:
            continue
        cut = int(np.floor(len(idx) * (1 - test_frac)))
        cut = max(1, min(cut, len(idx) - 1))
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    return np.array(train_idx), np.array(test_idx)


def cv_ridge_eval(df: pd.DataFrame, features: List[str], cv_scheme: str) -> Tuple[Dict, pd.Series, List[str], np.ndarray]:
    if not HAS_SKLEARN:
        return {"model_ok": 0, "error": "sklearn unavailable"}, pd.Series(dtype=float), [], np.array([])
    d = df.copy()
    y = pd.to_numeric(d["success"], errors="coerce")
    X = d[features].copy()
    keep = []
    for c in X.columns:
        vals = pd.to_numeric(X[c], errors="coerce")
        if vals.notna().sum() >= 20 and vals.nunique(dropna=True) > 1:
            X[c] = vals
            keep.append(c)
    X = X[keep]
    ok = y.notna()
    X = X.loc[ok]
    y = y.loc[ok]
    d_ok = d.loc[ok]
    if len(y) < 30 or X.shape[1] == 0:
        return {"model_ok": 0, "error": "too few rows/no features"}, pd.Series(dtype=float), keep, np.array([])

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 31)))
    ])

    preds = pd.Series(np.nan, index=X.index)

    fold_rows = []
    if cv_scheme == "subject_groupkfold":
        n_splits = min(5, d_ok["subject"].nunique())
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, d_ok["subject"])
        for fold, (tr, te) in enumerate(split_iter, 1):
            model.fit(X.iloc[tr], y.iloc[tr])
            pr = model.predict(X.iloc[te])
            preds.iloc[te] = pr
            fold_rows.append({"fold": fold, "n_train": len(tr), "n_test": len(te), "r2": r2_score(y.iloc[te], pr)})
    elif cv_scheme == "forward_within_subject":
        tr_idx, te_idx = cv_splits_forward_subject(d_ok)
        tr_pos = [X.index.get_loc(i) for i in tr_idx if i in X.index]
        te_pos = [X.index.get_loc(i) for i in te_idx if i in X.index]
        model.fit(X.iloc[tr_pos], y.iloc[tr_pos])
        pr = model.predict(X.iloc[te_pos])
        preds.iloc[te_pos] = pr
        fold_rows.append({"fold": 1, "n_train": len(tr_pos), "n_test": len(te_pos), "r2": r2_score(y.iloc[te_pos], pr)})
    else:
        raise ValueError(cv_scheme)

    valid = preds.notna()
    metrics = {
        "model_ok": 1,
        "n": int(valid.sum()),
        "n_features": int(X.shape[1]),
        "cv_scheme": cv_scheme,
        "r2_cv_overall": r2_score(y[valid], preds[valid]) if valid.sum() > 3 else np.nan,
        "mae_cv_overall": mean_absolute_error(y[valid], preds[valid]) if valid.sum() else np.nan,
        "rmse_cv_overall": math.sqrt(mean_squared_error(y[valid], preds[valid])) if valid.sum() else np.nan,
        "folds_json": json.dumps(fold_rows),
    }
    coefs = np.array([])
    try:
        model.fit(X, y)
        coefs = model.named_steps["ridge"].coef_
    except Exception:
        pass
    return metrics, preds, keep, coefs


def build_feature_sets(allowed: Dict[str, List[str]]) -> Dict[str, List[str]]:
    prev_behavior = allowed.get("previous_behavior", []) + allowed.get("time_order", [])
    prev_neural = allowed.get("previous_neural", [])
    preonset = allowed.get("preonset", [])
    domain = allowed.get("domain", [])

    sets = {
        "M1_previous_behavior_only": prev_behavior,
    }
    if prev_neural:
        sets["M2_previous_neural_only"] = prev_neural
        sets["M3_previous_behavior_plus_previous_neural"] = prev_behavior + prev_neural
    if preonset:
        sets["M4_preonset_roi_only"] = preonset
        sets["M5_previous_behavior_plus_preonset_roi"] = prev_behavior + preonset
    if domain:
        sets["M6_previous_behavior_plus_subject_domains"] = prev_behavior + domain
    if prev_neural and preonset:
        sets["M7_previous_behavior_plus_prev_neural_plus_preonset"] = prev_behavior + prev_neural + preonset
    return sets


def run_cv_all(df: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, imps, preds_all = [], [], []
    outcomes = {
        "all_reg_condition_blind": df,
        "up_only_condition_stratified": df[df["is_up"].astype(float) == 1],
        "down_only_condition_stratified": df[df["is_down"].astype(float) == 1],
    }
    for outcome, d in outcomes.items():
        for fs, cols in feature_sets.items():
            cols = [c for c in cols if c in d.columns]
            if not cols:
                continue
            for cv_scheme in ["subject_groupkfold", "forward_within_subject"]:
                metrics, preds, keep, coefs = cv_ridge_eval(d, cols, cv_scheme)
                metrics.update({"outcome": outcome, "feature_set": fs, "features": ",".join(keep)})
                rows.append(metrics)
                if len(preds):
                    prediction_rows = d[["subject", "run", "trial_in_run", "condition", "global_trial_index", "success"]].copy()
                    prediction_rows["outcome"] = outcome
                    prediction_rows["feature_set"] = fs
                    prediction_rows["cv_scheme"] = cv_scheme
                    prediction_rows["pred_success"] = preds
                    preds_all.append(prediction_rows)
                for c, coef in zip(keep, coefs):
                    imps.append({"outcome": outcome, "feature_set": fs, "cv_scheme": cv_scheme, "feature": c, "coef": float(coef), "abs_coef": float(abs(coef))})
    cv = pd.DataFrame(rows)
    imp = pd.DataFrame(imps).sort_values("abs_coef", ascending=False) if imps else pd.DataFrame()
    pred = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()
    return cv, imp, pred


def make_figures(outdir: Path, cv: pd.DataFrame, imp: pd.DataFrame):
    if not HAS_MPL:
        return
    mkdirp(outdir / "figures")
    if len(cv):
        plot = cv[cv["model_ok"] == 1].sort_values(["outcome", "cv_scheme", "r2_cv_overall"])
        labels = plot["outcome"] + " | " + plot["cv_scheme"] + " | " + plot["feature_set"]
        plt.figure(figsize=(11, max(5, 0.25 * len(plot))))
        plt.barh(range(len(plot)), plot["r2_cv_overall"])
        plt.yticks(range(len(plot)), labels, fontsize=7)
        plt.xlabel("Strict CV R²")
        plt.title("Strict previous-only prediction of regulation success")
        plt.tight_layout()
        plt.savefig(outdir / "figures" / "fig_01_strict_cv_r2.png", dpi=200)
        plt.close()
    if len(imp):
        plot = imp.head(30)
        plt.figure(figsize=(9, max(5, 0.25 * len(plot))))
        plt.barh(range(len(plot)), plot["abs_coef"])
        plt.yticks(range(len(plot)), plot["feature"], fontsize=7)
        plt.xlabel("|ridge coefficient|")
        plt.title("Top strict previous-only prediction features")
        plt.tight_layout()
        plt.savefig(outdir / "figures" / "fig_02_strict_feature_importance.png", dpi=200)
        plt.close()


def write_report(outdir: Path, manifest: Dict, audit: pd.DataFrame, cv: pd.DataFrame, mech: pd.DataFrame):
    lines = []
    lines.append("# Strict previous-only prediction of pain-regulation success - analysis ")
    lines.append("")
    lines.append("## Core rule")
    lines.append("For predicting trial t, no information from trial t itself is used as a predictor. Current condition, current temperature, current expected pain, current rating, current success, and current same-trial neural LSS/NPS/ROI values are excluded.")
    lines.append("")
    lines.append("")
    lines.append("## Dimensions")
    lines.append(f"- Regulation trials: **{manifest.get('n_reg_trials')}**")
    lines.append(f"- Subjects: **{manifest.get('n_subjects')}**")
    lines.append(f"- Previous-behavior features: **{manifest.get('n_previous_behavior_features')}**")
    lines.append(f"- Previous-neural features: **{manifest.get('n_previous_neural_features')}**")
    lines.append(f"- Pre-onset ROI features: **{manifest.get('n_preonset_features')}**")
    lines.append(f"- Subject-domain features: **{manifest.get('n_domain_features')}**")
    lines.append("")
    lines.append("## Feature audit")
    if len(audit):
        lines.append(audit.groupby(["source", "allowed", "reason"]).size().reset_index(name="count").to_markdown(index=False))
    else:
        lines.append("No audit rows.")
    lines.append("")
    lines.append("## Cross-validated strict prediction")
    if len(cv):
        show = cv.sort_values(["outcome", "cv_scheme", "r2_cv_overall"], ascending=[True, True, False])
        lines.append(show[["outcome", "feature_set", "cv_scheme", "n", "n_features", "r2_cv_overall", "mae_cv_overall", "rmse_cv_overall"]].to_markdown(index=False))
    else:
        lines.append("No CV output.")
    lines.append("")
    lines.append("## Mechanism/statistical tests")
    if len(mech):
        show = mech[(mech["term"].astype(str) != "Intercept") & (~mech["term"].astype(str).str.startswith("C(subject)")) & (mech["p"].notna())].sort_values("p").head(40)
        if len(show):
            lines.append(show[["feature_source", "model_name", "term", "beta", "t_or_z", "p", "q_fdr_target_terms", "n", "r2"]].to_markdown(index=False))
        else:
            lines.append("No target terms.")
    else:
        lines.append("No mechanism tests.")
    lines.append("")
    lines.append("## Interpretation guardrails")
    lines.append("- `all_reg_condition_blind` is the strictest prediction; it does not use current condition.")
    lines.append("- `up_only` and `down_only` are condition-stratified evaluations; condition is used only to select rows, not as a predictor.")
    lines.append("- Previous-neural predictors are from trial t-1 or earlier.")
    lines.append("- Pre-onset ROI predictors are from BOLD samples before onset_t minus safety gap; they contain no post-onset information.")
    lines.append("- Subject-domain predictors are trait/individual-difference predictors, not trial-level neural states.")
    (outdir / "strict_previous_only_success_prediction_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derivatives", required=True)
    ap.add_argument("--trialwise", default=None)
    ap.add_argument("--previous-neural-table", default=None)
    ap.add_argument("--preonset-table", default=None)
    ap.add_argument("--domain-scores", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-preonset", action="store_true")
    ap.add_argument("--no-domains", action="store_true")
    ap.add_argument("--top-n-tests", type=int, default=30)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    derivatives = Path(args.derivatives)
    outdir = Path(args.outdir) if args.outdir else derivatives / "strict_previous_only_success_prediction_analysis"
    mkdirp(outdir / "tables")

    trialwise_path = Path(args.trialwise) if args.trialwise else infer_trialwise_path(derivatives)
    if trialwise_path is None or not trialwise_path.exists():
        raise FileNotFoundError("Could not find trialwise table.")

    reg, info = make_previous_behavior_features(safe_read_tsv(trialwise_path))

    prev_neural_path = Path(args.previous_neural_table) if args.previous_neural_table else infer_prev_neural_table(derivatives)
    reg, prev_neural_cols = attach_previous_neural(reg, prev_neural_path)

    preonset_path = None if args.no_preonset else (Path(args.preonset_table) if args.preonset_table else infer_preonset_table(derivatives))
    reg, preonset_cols = attach_preonset(reg, preonset_path)

    domain_path = None if args.no_domains else (Path(args.domain_scores) if args.domain_scores else infer_domain_scores(derivatives))
    reg, domain_cols = attach_domains(reg, domain_path)

    audit, allowed = build_feature_audit(reg, prev_neural_cols, preonset_cols, domain_cols)
    feature_sets = build_feature_sets(allowed)

    cv, imp, preds = run_cv_all(reg, feature_sets)
    mech = run_mechanism_tests(reg, allowed, top_n=args.top_n_tests)

    reg.to_csv(outdir / "tables" / "strict_previous_only_state_table.tsv", sep="\t", index=False)
    audit.to_csv(outdir / "tables" / "strict_feature_audit.tsv", sep="\t", index=False)
    cv.to_csv(outdir / "tables" / "strict_prediction_cv.tsv", sep="\t", index=False)
    imp.to_csv(outdir / "tables" / "strict_feature_importance_ridge.tsv", sep="\t", index=False)
    preds.to_csv(outdir / "tables" / "strict_cv_predictions.tsv", sep="\t", index=False)
    mech.to_csv(outdir / "tables" / "strict_mechanism_tests.tsv", sep="\t", index=False)

    if not args.no_figures:
        make_figures(outdir, cv, imp)

    manifest = {
        "derivatives": str(derivatives),
        "trialwise": str(trialwise_path),
        "previous_neural_table": str(prev_neural_path) if prev_neural_path else None,
        "preonset_table": str(preonset_path) if preonset_path else None,
        "domain_scores": str(domain_path) if domain_path else None,
        "outdir": str(outdir),
        "n_reg_trials": int(len(reg)),
        "n_subjects": int(reg["subject"].nunique()),
        "n_previous_behavior_features": int(len(allowed.get("previous_behavior", []))),
        "n_time_order_features": int(len(allowed.get("time_order", []))),
        "n_previous_neural_features": int(len(allowed.get("previous_neural", []))),
        "n_preonset_features": int(len(allowed.get("preonset", []))),
        "n_domain_features": int(len(allowed.get("domain", []))),
        "strict_rule": "For prediction of trial t, the model excludes current-trial condition, temperature, expected pain, rating, success, and same-trial neural features. Previous-neural inputs are filtered strictly, empty pre-onset files are skipped, and run-relative pre-onset times are used to prevent source-prefix leakage.",
        "condition_use": "Current condition is used only for post-hoc stratified evaluation of up-only and down-only models, not as a predictor.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(outdir, manifest, audit, cv, mech)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
