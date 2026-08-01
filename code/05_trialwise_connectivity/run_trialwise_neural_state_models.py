#!/usr/bin/env python3
"""
run_trialwise_neural_state_models.py

Goal
----
Integrate trial-wise neural state with behavioral/computational pain-regulation
state in ds000140.

Pipeline context:
- computational_pain_regulation_analysis/
- trialwise_state_success_prediction_analysis

Main scientific questions:
1) What happens in the brain during successful vs unsuccessful up/down regulation?
2) Are up and down supported by different neural mechanisms?
3) Does neural state add predictive value beyond behavioral state/history?
4) Can previous neural state predict the next/current regulation success?
5) Which effects are prospective and which are same-trial explanatory?

Important distinction:
- previous-trial neural features: closer to prospective/state prediction
- current-trial neural features: explanatory/concurrent mechanism, not pre-trial prediction

Outputs:
- tables/neural_file_inventory.tsv
- tables/trialwise_neural_state_table.tsv
- tables/neural_feature_qc.tsv
- tables/neural_mechanism_interaction_tests.tsv
- tables/up_neural_success_models.tsv
- tables/down_neural_success_models.tsv
- tables/prospective_neural_prediction_cv.tsv
- tables/feature_importance_ridge_neural.tsv
- tables/cv_neural_predictions.tsv
- trialwise_neural_state_integration_report.md
- manifest.json

Analysis workflow for the ds000140 pain-regulation project
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

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
    from sklearn.model_selection import GroupKFold, KFold
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


# ---------------------------
# Basic utilities
# ---------------------------

def mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_read_tsv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return None


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


def sanitize_feature_name(name: str) -> str:
    s = str(name)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "feature"
    if re.match(r"^[0-9]", s):
        s = "f_" + s
    return s[:120]


def infer_trialwise_path(derivatives: Path) -> Optional[Path]:
    candidates = [
        derivatives / "computational_pain_regulation_analysis" / "tables" / "trialwise_model_predictions.tsv",
        derivatives / "trialwise_model_predictions.tsv",
        derivatives / "trialwise_state_success_prediction_analysis" / "tables" / "trialwise_state_table.tsv",
        derivatives / "trialwise_state_success_prediction_analysis" / "tables" / "trialwise_state_table.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    found = list(derivatives.glob("**/trialwise_model_predictions.tsv"))
    return found[0] if found else None


def infer_domain_scores_path(derivatives: Path) -> Optional[Path]:
    candidates = [
        derivatives / "refined_individual_fingerprint_analysis" / "tables" / "domain_scores.tsv",
        derivatives / "domain_scores.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    found = list(derivatives.glob("**/domain_scores.tsv"))
    return found[0] if found else None


def candidate_neural_files(derivatives: Path) -> List[Path]:
    """
    Conservative recursive search for trial-level neural tables.
    It intentionally includes many candidate names, then later filters by mergeability.
    """
    patterns = [
        "**/*nps*.tsv",
        "**/*NPS*.tsv",
        "**/*lss*.tsv",
        "**/*LSS*.tsv",
        "**/*roi*value*.tsv",
        "**/*ROI*value*.tsv",
        "**/*roi*lss*.tsv",
        "**/*ROI*LSS*.tsv",
        "**/*trialwise*roi*.tsv",
        "**/*trialwise*ROI*.tsv",
        "**/*trialwise*signature*.tsv",
        "**/*signature*value*.tsv",
        "**/*fc*trial*.tsv",
        "**/*FC*trial*.tsv",
        "**/*ec*trial*.tsv",
        "**/*EC*trial*.tsv",
        "**/*edge*trial*.tsv",
        "**/*pathway*trial*.tsv",
    ]
    files = []
    for pat in patterns:
        files.extend(list(derivatives.glob(pat)))
    # Exclude large non-tabular files and summary outputs that are not trialwise unless no alternative is available.
    uniq = []
    seen = set()
    for p in files:
        if p in seen:
            continue
        seen.add(p)
        if p.name.startswith("."):
            continue
        if "report" in p.name.lower():
            continue
        if "manifest" in p.name.lower():
            continue
        if p.suffix.lower() != ".tsv":
            continue
        uniq.append(p)
    return uniq


def pick_success_col(df: pd.DataFrame) -> str:
    for c in ["reg_success_z_model", "reg_success_z", "success", "reg_success_raw_model", "reg_success_raw"]:
        if c in df.columns:
            return c
    raise ValueError("No success column found. Expected reg_success_z_model/reg_success_z.")


def pick_expected_col(df: pd.DataFrame) -> str:
    for c in ["expected_pain_z_passive_model", "expected_pain_z", "expected_pain", "expected_pain_raw_passive_model", "expected_pain_raw"]:
        if c in df.columns:
            return c
    raise ValueError("No expected passive pain column found.")


def ensure_condition_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "condition" not in out.columns:
        # Derive the value from status flags when needed.
        if "is_up" in out.columns and "is_down" in out.columns:
            out["condition"] = np.where(out["is_up"].astype(float) == 1, "up",
                                        np.where(out["is_down"].astype(float) == 1, "down", "passive"))
        else:
            out["condition"] = "unknown"
    out["condition"] = out["condition"].astype(str).str.lower()
    if "is_up" not in out.columns:
        out["is_up"] = out["condition"].eq("up").astype(int)
    if "is_down" not in out.columns:
        out["is_down"] = out["condition"].eq("down").astype(int)
    if "is_passive" not in out.columns:
        out["is_passive"] = out["condition"].isin(["passive", "view", "baseline"]).astype(int)
    if "is_reg" not in out.columns:
        out["is_reg"] = (out["is_up"].astype(float).eq(1) | out["is_down"].astype(float).eq(1)).astype(int)
    return out


def harmonize_core_trial_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Subject
    for c in ["sub", "participant_id", "subject_id"]:
        if c in out.columns and "subject" not in out.columns:
            out = out.rename(columns={c: "subject"})
    if "subject" in out.columns:
        out["subject"] = out["subject"].astype(str).str.replace("^sub-", "", regex=True)
        out["subject"] = out["subject"].apply(lambda x: f"sub-{x}" if not str(x).startswith("sub-") else str(x))
    # Run
    if "run_label" in out.columns and "run" not in out.columns:
        out["run"] = pd.to_numeric(out["run_label"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    for c in ["run_id", "run_index", "run_number"]:
        if c in out.columns and "run" not in out.columns:
            out["run"] = pd.to_numeric(out[c], errors="coerce")
    if "run" in out.columns:
        out["run"] = pd.to_numeric(out["run"], errors="coerce")
    # Trial
    trial_aliases = ["trial_in_run", "trial_lss_index", "trial_index", "trial_num", "trial", "trial_number", "event_index"]
    if "trial_in_run" not in out.columns:
        for c in trial_aliases:
            if c in out.columns:
                out = out.rename(columns={c: "trial_in_run"})
                break
    if "trial_in_run" in out.columns:
        out["trial_in_run"] = pd.to_numeric(out["trial_in_run"], errors="coerce")
    # Condition
    if "trial_type" in out.columns and "condition" not in out.columns:
        out = out.rename(columns={"trial_type": "condition"})
    if "condition" in out.columns:
        out["condition"] = out["condition"].astype(str).str.lower()
    return out


def make_behavior_state_table(trialwise: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = harmonize_core_trial_columns(trialwise)
    df = ensure_condition_flags(df)

    success_col = pick_success_col(df)
    expected_col = pick_expected_col(df)

    df["success"] = pd.to_numeric(df[success_col], errors="coerce")
    df["expected_pain"] = pd.to_numeric(df[expected_col], errors="coerce")

    # Need an order variable for lagging.
    if "global_trial_index" not in df.columns:
        if "global_trial_z_subject" in df.columns:
            # Not ideal but gives stable sort if no index exists.
            df["global_trial_index"] = df.groupby("subject")["global_trial_z_subject"].rank(method="first")
        elif "trial_in_run" in df.columns and "run" in df.columns:
            df["global_trial_index"] = df["run"].fillna(0) * 1000 + df["trial_in_run"].fillna(0)
        else:
            df["global_trial_index"] = df.groupby("subject").cumcount() + 1

    df = df.sort_values(["subject", "global_trial_index"]).copy()

    # Create/center common behavior variables.
    numeric_candidates = [
        "rating_z_subject", "rating_raw", "temperature_z_subject", "temperature",
        "global_trial_z_subject", "up_first", "expected_pain", "success"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Previous behavioral states.
    lag_cols = {
        "rating_z_subject": "prev_rating_z",
        "rating_raw": "prev_rating_raw",
        "temperature_z_subject": "prev_temperature_z",
        "temperature": "prev_temperature",
        "expected_pain": "prev_expected_pain",
        "success": "prev_success_any",
        "is_up": "prev_is_up",
        "is_down": "prev_is_down",
        "is_passive": "prev_is_passive",
    }
    for c, new in lag_cols.items():
        if c in df.columns:
            df[new] = df.groupby("subject")[c].shift(1)

    # Previous regulation success only.
    df["prev_reg_success"] = np.nan
    df["prev_reg_condition_is_up"] = np.nan
    for sub, g in df.groupby("subject", sort=False):
        last_success = np.nan
        last_is_up = np.nan
        vals_success = []
        vals_up = []
        for _, row in g.iterrows():
            vals_success.append(last_success)
            vals_up.append(last_is_up)
            if int(row.get("is_reg", 0)) == 1 and pd.notna(row.get("success", np.nan)):
                last_success = row.get("success", np.nan)
                last_is_up = row.get("is_up", np.nan)
        df.loc[g.index, "prev_reg_success"] = vals_success
        df.loc[g.index, "prev_reg_condition_is_up"] = vals_up

    # Cumulative behavioral states before current trial.
    for c, new in [
        ("rating_z_subject", "cum_prev_rating_mean"),
        ("expected_pain", "cum_prev_expected_mean"),
        ("success", "cum_prev_success_mean"),
    ]:
        if c in df.columns:
            df[new] = df.groupby("subject")[c].transform(lambda s: s.expanding().mean().shift(1))

    if "is_passive" in df.columns and "rating_z_subject" in df.columns:
        df["_passive_rating_z_working"] = np.where(df["is_passive"].astype(float) == 1, df["rating_z_subject"], np.nan)
        df["cum_prev_passive_rating_mean"] = df.groupby("subject")["_passive_rating_z_working"].transform(lambda s: s.expanding().mean().shift(1))
        df.drop(columns=["_passive_rating_z_working"], inplace=True)

    if "run" in df.columns and "rating_z_subject" in df.columns:
        df["run_prev_rating_mean"] = df.groupby(["subject", "run"])["rating_z_subject"].transform(lambda s: s.expanding().mean().shift(1))

    # Keep regulation trials with success.
    reg = df[df["is_reg"].astype(float) == 1].copy()

    # Center variables globally across regulation trials.
    for c in [
        "is_up", "is_down", "expected_pain", "temperature_z_subject", "temperature",
        "global_trial_z_subject", "up_first",
        "prev_rating_z", "prev_rating_raw", "prev_temperature_z", "prev_temperature",
        "prev_expected_pain", "prev_success_any", "prev_reg_success",
        "cum_prev_rating_mean", "cum_prev_expected_mean", "cum_prev_success_mean",
        "cum_prev_passive_rating_mean", "run_prev_rating_mean"
    ]:
        if c in reg.columns:
            vals = pd.to_numeric(reg[c], errors="coerce")
            reg[c + "_c"] = vals - vals.mean()

    info = {"success_col": success_col, "expected_col": expected_col}
    return reg, info


# ---------------------------
# Neural table preparation
# ---------------------------

META_COL_KEYWORDS = [
    "subject", "run", "trial", "condition", "onset", "duration", "filename",
    "path", "file", "roi", "region", "source", "target", "edge", "network",
    "contrast", "model", "fold", "session", "task"
]


LEAK_OR_NON_NEURAL_PATTERNS = [
    "rating", "success", "expected", "observed", "response",
    "temperature", "temp_", "stim", "heat",
    "onset", "duration", "trial_", "global_trial", "run_",
    "condition", "design", "rank", "cols", "n_vol", "motion", "fd", "dvars",
    "resid", "residual", "error", "prediction", "pred_", "model_r2",
    "subject", "intercept"
]

def is_valid_neural_feature_name(col: str) -> bool:
    """
    .1 leakage filter.

    Keep only genuine neural features:
    - ROI LSS/beta columns with specific ROI names from wide/pivoted tables.
    - NPS/signature expression columns.

    Exclude behavioral/outcome/design columns that leaked into neural tables:
    rating_for_model, expected pain, temperature, onset, design rank/cols,
    n_vols, trial/run variables, etc.
    """
    lc = str(col).lower()

    if any(pat in lc for pat in LEAK_OR_NON_NEURAL_PATTERNS):
        return False

    # NPS/signature features.
    if "nps" in lc:
        # Keep actual NPS expression/norm; exclude design/QC handled above.
        return any(k in lc for k in ["dot", "l2norm", "weighted", "expression", "score"])

    if "signature" in lc:
        return True

    # ROI features: require specific ROI pattern from wide table or pivoted long table.
    # Exclude a generic long-table roi_beta field because duplicate removal can retain an arbitrary ROI.
    if "_roi_" in lc:
        return True
    if "_roi_" in lc and "lss_wide" in lc:
        return True

    # FC/EC/edge features, if trialwise edge tables exist.
    if any(k in lc for k in ["_edge_", "_to_", "fc_", "ec_", "connectivity"]):
        return True

    return False

def detect_numeric_feature_columns(df: pd.DataFrame, key_cols: List[str]) -> List[str]:
    feats = []
    for c in df.columns:
        if c in key_cols:
            continue
        lc = c.lower()
        if any(k == lc for k in ["subject", "condition"]):
            continue
        if lc in ["run", "trial_in_run"]:
            continue
        if any(lc.startswith(k + "_") for k in ["subject", "condition"]):
            continue
        if not is_valid_neural_feature_name(c):
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() >= 10 and vals.nunique(dropna=True) > 1:
            feats.append(c)
    return feats


def table_is_trialwise(df: pd.DataFrame) -> bool:
    has_sub = "subject" in df.columns
    has_run = "run" in df.columns
    has_trial = "trial_in_run" in df.columns or "global_trial_index" in df.columns
    return bool(has_sub and (has_trial or has_run) and len(df) > 100)


def long_to_wide_if_needed(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = df.copy()
    key_base = [k for k in ["subject", "run", "trial_in_run", "global_trial_index", "condition"] if k in out.columns]

    # ROI long format: ROI or region identifier with a numeric value column.
    roi_col = None
    for c in ["roi", "ROI", "region", "region_name", "mask", "node"]:
        if c in out.columns:
            roi_col = c
            break

    value_col = None
    for c in ["value", "beta", "activity", "signal", "lss_beta", "estimate", "score", "dot", "pattern_expression"]:
        if c in out.columns and pd.to_numeric(out[c], errors="coerce").notna().sum() > 10:
            value_col = c
            break

    if roi_col and value_col and len(key_base) >= 3:
        roi_long = out[key_base + [roi_col, value_col]].copy()
        roi_long[roi_col] = roi_long[roi_col].astype(str).map(sanitize_feature_name)
        wide = roi_long.pivot_table(index=key_base, columns=roi_col, values=value_col, aggfunc="mean").reset_index()
        wide.columns = [str(c) if not isinstance(c, tuple) else "_".join([str(x) for x in c if x]) for c in wide.columns]
        rename = {c: f"{sanitize_feature_name(source_name)}_roi_{sanitize_feature_name(c)}" for c in wide.columns if c not in key_base}
        return wide.rename(columns=rename)

    # Edge long format: source + target + value.
    source_col = next((c for c in ["source", "src", "from", "node_from", "roi1"] if c in out.columns), None)
    target_col = next((c for c in ["target", "tgt", "to", "node_to", "roi2"] if c in out.columns), None)
    edge_val_col = value_col
    if source_col and target_col and edge_val_col and len(key_base) >= 3:
        edge_long = out[key_base + [source_col, target_col, edge_val_col]].copy()
        edge_long["_edge_name"] = edge_long[source_col].astype(str).map(sanitize_feature_name) + "_to_" + edge_long[target_col].astype(str).map(sanitize_feature_name)
        wide = edge_long.pivot_table(index=key_base, columns="_edge_name", values=edge_val_col, aggfunc="mean").reset_index()
        rename = {c: f"{sanitize_feature_name(source_name)}_edge_{sanitize_feature_name(c)}" for c in wide.columns if c not in key_base}
        return wide.rename(columns=rename)

    return out


def prepare_neural_table(path: Path, max_features_per_file: int = 1000) -> Tuple[Optional[pd.DataFrame], Dict]:
    raw = safe_read_tsv(path)
    info = {
        "path": str(path),
        "file": path.name,
        "loaded": raw is not None,
        "n_rows_raw": 0,
        "n_cols_raw": 0,
        "is_trialwise": False,
        "n_features": 0,
        "merge_keys": "",
        "status": "not_loaded",
    }
    if raw is None:
        return None, info

    info["n_rows_raw"] = int(len(raw))
    info["n_cols_raw"] = int(raw.shape[1])

    df = harmonize_core_trial_columns(raw)
    df = ensure_condition_flags(df) if "condition" in df.columns else df
    df = long_to_wide_if_needed(df, path.stem)

    # Harmonize again after pivot.
    df = harmonize_core_trial_columns(df)

    if not table_is_trialwise(df):
        info["status"] = "not_trialwise_or_missing_keys"
        return None, info

    merge_keys = [k for k in ["subject", "run", "trial_in_run"] if k in df.columns]
    if len(merge_keys) < 3:
        info["status"] = "insufficient_merge_keys"
        return None, info

    key_cols = merge_keys + (["condition"] if "condition" in df.columns else [])
    feat_cols = detect_numeric_feature_columns(df, key_cols)

    # Exclude original numeric key-like fields.
    feat_cols = [c for c in feat_cols if c not in ["run", "trial_in_run", "global_trial_index", "is_up", "is_down", "is_reg", "is_passive"]]

    if len(feat_cols) == 0:
        info["status"] = "no_valid_neural_features_after_leakage_filter"
        return None, info

    # Limit memory use by retaining the highest-variance features when the feature threshold is exceeded.
    if len(feat_cols) > max_features_per_file:
        variances = df[feat_cols].apply(pd.to_numeric, errors="coerce").var(axis=0, ddof=0).sort_values(ascending=False)
        feat_cols = list(variances.head(max_features_per_file).index)

    prefix = sanitize_feature_name(path.parent.name + "_" + path.stem)
    rename = {}
    for c in feat_cols:
        if not str(c).startswith(prefix):
            rename[c] = f"{prefix}_{sanitize_feature_name(c)}"
    keep_cols = merge_keys + feat_cols
    if "condition" in df.columns:
        # Do not merge on condition by default, but keep for QC.
        keep_cols = merge_keys + ["condition"] + feat_cols

    out = df[keep_cols].copy()
    out = out.rename(columns=rename)
    out = out.drop_duplicates(merge_keys)

    info["is_trialwise"] = True
    info["n_features"] = int(len(feat_cols))
    info["merge_keys"] = ",".join(merge_keys)
    info["status"] = "usable"
    return out, info


def merge_neural_tables(reg: pd.DataFrame, neural_paths: List[Path], max_features_per_file: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df = reg.copy()
    inventory_rows = []
    neural_cols_all = []

    for path in neural_paths:
        nt, info = prepare_neural_table(path, max_features_per_file=max_features_per_file)
        inventory_rows.append(info)
        if nt is None:
            continue

        merge_keys = [k for k in ["subject", "run", "trial_in_run"] if k in df.columns and k in nt.columns]
        if len(merge_keys) < 3:
            inventory_rows[-1]["status"] = "merge_failed_keys"
            continue

        # Do not merge on condition unless needed; trial index should identify.
        feat_cols = [c for c in nt.columns if c not in merge_keys + ["condition"]]
        before = len(df)
        df = df.merge(nt[merge_keys + feat_cols], on=merge_keys, how="left")
        after = len(df)
        assert before == after

        for c in feat_cols:
            if c in df.columns:
                nonmiss = int(df[c].notna().sum())
                if nonmiss >= 20:
                    neural_cols_all.append(c)
        inventory_rows[-1]["merged_features_nonmissing_ge20"] = int(sum(df[c].notna().sum() >= 20 for c in feat_cols if c in df.columns))

    inventory = pd.DataFrame(inventory_rows)
    neural_cols_all = list(dict.fromkeys(neural_cols_all))
    return df, inventory, neural_cols_all


def attach_domain_scores(reg: pd.DataFrame, domain_path: Optional[Path]) -> Tuple[pd.DataFrame, List[str]]:
    df = reg.copy()
    if domain_path is None or not domain_path.exists():
        return df, []
    dom = safe_read_tsv(domain_path)
    if dom is None or "subject" not in dom.columns:
        return df, []
    dom = harmonize_core_trial_columns(dom)
    domain_cols = []
    for c in dom.columns:
        if c == "subject":
            continue
        vals = pd.to_numeric(dom[c], errors="coerce")
        if vals.notna().sum() >= 5 and vals.nunique(dropna=True) > 1:
            dom[c] = vals
            domain_cols.append(c)
    # Standardize participant-level rows.
    for c in domain_cols:
        dom[c + "_z"] = zscore(dom[c])
    use_cols = ["subject"] + [c + "_z" for c in domain_cols]
    df = df.merge(dom[use_cols], on="subject", how="left")
    return df, [c + "_z" for c in domain_cols]


def create_neural_lag_features(df: pd.DataFrame, neural_cols: List[str], max_lag_cols: int = 500) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Create previous-trial versions of neural features. To limit dimensionality, retain at most max_lag_cols columns with the highest coverage and variance.
    """
    out = df.sort_values(["subject", "global_trial_index"] if "global_trial_index" in df.columns else ["subject", "run", "trial_in_run"]).copy()
    qc_rows = []

    # QC and standardize current neural features.
    usable = []
    for c in neural_cols:
        vals = pd.to_numeric(out[c], errors="coerce")
        coverage = vals.notna().mean()
        var = vals.var(ddof=0)
        qc_rows.append({"feature": c, "coverage": coverage, "variance": var, "n_nonmissing": int(vals.notna().sum())})
        if vals.notna().sum() >= 30 and np.isfinite(var) and var > 0:
            out[c] = vals
            usable.append(c)

    qc = pd.DataFrame(qc_rows).sort_values(["coverage", "variance"], ascending=False) if qc_rows else pd.DataFrame()
    if len(usable) > max_lag_cols:
        keep = list(qc[qc["feature"].isin(usable)].head(max_lag_cols)["feature"])
        usable = keep

    current_z_cols = []
    prev_cols = []
    for c in usable:
        cz = c + "_current_z"
        pz = c + "_prev_z"
        out[cz] = zscore(out[c])
        out[pz] = out.groupby("subject")[cz].shift(1)
        current_z_cols.append(cz)
        prev_cols.append(pz)

    return out, current_z_cols + prev_cols, qc


# ---------------------------
# Statistics
# ---------------------------

def cluster_ols(formula: str, data: pd.DataFrame, cluster_col: str = "subject") -> pd.DataFrame:
    if not HAS_STATSMODELS:
        return pd.DataFrame([{"model_ok": 0, "error": "statsmodels unavailable", "formula": formula}])
    d = data.copy()

    rhs = formula.replace("~", "+").replace("*", "+").replace(":", "+")
    rhs = rhs.replace("C(subject)", "subject")
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rhs)
    needed = []
    for tok in tokens:
        if tok in d.columns and tok not in needed:
            needed.append(tok)
    if "success" in d.columns and "success" not in needed:
        needed = ["success"] + needed
    if cluster_col in d.columns and cluster_col not in needed:
        needed.append(cluster_col)

    for c in needed:
        if c != cluster_col and c in d.columns and not pd.api.types.is_numeric_dtype(d[c]):
            d[c] = pd.to_numeric(d[c], errors="ignore")
    d_fit = d.dropna(subset=[c for c in needed if c in d.columns]).copy()

    try:
        if len(d_fit) < 20:
            raise ValueError(f"too few rows after NA drop: {len(d_fit)}")
        model = smf.ols(formula, data=d_fit).fit(
            cov_type="cluster",
            cov_kwds={"groups": d_fit[cluster_col].values}
        )
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
        return pd.DataFrame([{
            "formula": formula,
            "term": "_MODEL_FAILED_",
            "beta": np.nan, "se": np.nan, "t_or_z": np.nan, "p": np.nan,
            "n": len(d_fit), "r2": np.nan, "model_ok": 0, "error": str(e)
        }])


def screen_features_by_correlation(df: pd.DataFrame, features: List[str], outcome: str = "success", top_n: int = 50) -> pd.DataFrame:
    rows = []
    for c in features:
        if c not in df.columns:
            continue
        valid_rows = df[[c, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(valid_rows) < 30 or valid_rows[c].nunique() < 3:
            continue
        if HAS_SCIPY:
            r, p = stats.pearsonr(valid_rows[c], valid_rows[outcome])
        else:
            r = valid_rows[c].corr(valid_rows[outcome])
            p = np.nan
        rows.append({"feature": c, "r": r, "abs_r": abs(r), "p": p, "n": len(valid_rows)})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_fdr"] = fdr_bh(out["p"].tolist())
        out = out.sort_values("abs_r", ascending=False).head(top_n)
    return out


def run_neural_mechanism_tests(df: pd.DataFrame, current_cols: List[str], prev_cols: List[str], domain_cols: List[str], top_n: int = 40) -> pd.DataFrame:
    rows = []

    base_terms = [t for t in [
        "is_up_c", "expected_pain_c", "is_up_c:expected_pain_c",
        "global_trial_z_subject_c", "up_first_c", "temperature_z_subject_c",
        "prev_rating_z_c", "prev_reg_success_c", "cum_prev_passive_rating_mean_c"
    ] if t in df.columns]

    # Always include behavior-only dynamic baseline.
    formula = "success ~ " + " + ".join(base_terms) + " + C(subject)"
    out = cluster_ols(formula, df, "subject")
    out.insert(0, "model_name", "behavior_dynamic_subject_FE")
    out.insert(1, "feature_type", "behavior")
    rows.append(out)

    # Screen neural columns first to keep models finite.
    current_screen = screen_features_by_correlation(df, current_cols, "success", top_n=top_n)
    prev_screen = screen_features_by_correlation(df, prev_cols, "success", top_n=top_n)
    domain_screen = screen_features_by_correlation(df, domain_cols, "success", top_n=min(top_n, len(domain_cols)))

    # Current trial neural explanatory models.
    for c in current_screen.get("feature", []):
        formula = f"success ~ is_up_c*{c} + expected_pain_c + global_trial_z_subject_c + up_first_c + temperature_z_subject_c + prev_rating_z_c + prev_reg_success_c + C(subject)"
        out = cluster_ols(formula, df, "subject")
        out.insert(0, "model_name", f"current_neural_condition_{c}")
        out.insert(1, "feature_type", "current_trial_neural_explanatory")
        rows.append(out)

        formula2 = f"success ~ is_up_c*expected_pain_c*{c} + global_trial_z_subject_c + up_first_c + temperature_z_subject_c + prev_rating_z_c + prev_reg_success_c + C(subject)"
        out2 = cluster_ols(formula2, df, "subject")
        out2.insert(0, "model_name", f"current_neural_state_condition_{c}")
        out2.insert(1, "feature_type", "current_trial_neural_explanatory")
        rows.append(out2)

    # Previous-trial neural prospective models.
    for c in prev_screen.get("feature", []):
        formula = f"success ~ is_up_c*{c} + expected_pain_c + global_trial_z_subject_c + up_first_c + temperature_z_subject_c + prev_rating_z_c + prev_reg_success_c + C(subject)"
        out = cluster_ols(formula, df, "subject")
        out.insert(0, "model_name", f"prev_neural_condition_{c}")
        out.insert(1, "feature_type", "previous_trial_neural_prospective")
        rows.append(out)

        formula2 = f"success ~ is_up_c*expected_pain_c*{c} + global_trial_z_subject_c + up_first_c + temperature_z_subject_c + prev_rating_z_c + prev_reg_success_c + C(subject)"
        out2 = cluster_ols(formula2, df, "subject")
        out2.insert(0, "model_name", f"prev_neural_state_condition_{c}")
        out2.insert(1, "feature_type", "previous_trial_neural_prospective")
        rows.append(out2)

    # Subject-level domains: no subject FE because collinear.
    for c in domain_screen.get("feature", []):
        formula = f"success ~ is_up_c*{c} + expected_pain_c + global_trial_z_subject_c + up_first_c + temperature_z_subject_c + prev_rating_z_c + prev_reg_success_c"
        out = cluster_ols(formula, df, "subject")
        out.insert(0, "model_name", f"domain_condition_{c}")
        out.insert(1, "feature_type", "subject_level_domain")
        rows.append(out)

        formula2 = f"success ~ is_up_c*expected_pain_c*{c} + global_trial_z_subject_c + up_first_c + temperature_z_subject_c + prev_rating_z_c + prev_reg_success_c"
        out2 = cluster_ols(formula2, df, "subject")
        out2.insert(0, "model_name", f"domain_state_condition_{c}")
        out2.insert(1, "feature_type", "subject_level_domain")
        rows.append(out2)

    res = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(res):
        res["is_subject_dummy"] = res["term"].astype(str).str.startswith("C(subject)")
        res["is_target_term"] = (~res["is_subject_dummy"]) & (res["term"].astype(str) != "Intercept") & res["p"].notna()
        res["q_fdr_target_terms"] = np.nan
        mask = res["is_target_term"]
        res.loc[mask, "q_fdr_target_terms"] = fdr_bh(res.loc[mask, "p"].tolist())
    return res


def run_condition_specific_neural_models(df: pd.DataFrame, current_cols: List[str], prev_cols: List[str], top_n: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outputs = {}
    for cond_name, sub in [("up", df[df["is_up"].astype(float) == 1]), ("down", df[df["is_down"].astype(float) == 1])]:
        rows = []
        feature_pool = current_cols + prev_cols
        screen = screen_features_by_correlation(sub, feature_pool, "success", top_n=top_n)
        base_terms = [t for t in [
            "expected_pain_c", "global_trial_z_subject_c", "up_first_c",
            "temperature_z_subject_c", "prev_rating_z_c", "prev_reg_success_c",
            "cum_prev_passive_rating_mean_c"
        ] if t in sub.columns]
        # Behavior-only model.
        if base_terms:
            formula = "success ~ " + " + ".join(base_terms) + " + C(subject)"
            out = cluster_ols(formula, sub, "subject")
            out.insert(0, "condition_model", cond_name)
            out.insert(1, "feature_tested", "_behavior_only_")
            rows.append(out)
        for c in screen.get("feature", []):
            formula = f"success ~ {c} + " + " + ".join(base_terms) + " + C(subject)"
            out = cluster_ols(formula, sub, "subject")
            out.insert(0, "condition_model", cond_name)
            out.insert(1, "feature_tested", c)
            rows.append(out)
        res = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if len(res):
            res["is_target_term"] = (~res["term"].astype(str).str.startswith("C(subject)")) & (res["term"].astype(str) != "Intercept") & res["p"].notna()
            res["q_fdr_target_terms"] = np.nan
            mask = res["is_target_term"]
            res.loc[mask, "q_fdr_target_terms"] = fdr_bh(res.loc[mask, "p"].tolist())
        outputs[cond_name] = res
    return outputs.get("up", pd.DataFrame()), outputs.get("down", pd.DataFrame())


# ---------------------------
# CV prediction
# ---------------------------

def make_feature_sets(df: pd.DataFrame, current_cols: List[str], prev_cols: List[str], domain_cols: List[str], max_neural: int = 100) -> Dict[str, List[str]]:
    behavior_state = [c for c in [
        "is_up", "expected_pain", "temperature_z_subject", "temperature", "global_trial_z_subject", "up_first"
    ] if c in df.columns]
    behavior_history = behavior_state + [c for c in [
        "prev_rating_z", "prev_rating_raw", "prev_temperature_z", "prev_temperature",
        "prev_expected_pain", "prev_reg_success",
        "cum_prev_rating_mean", "cum_prev_expected_mean", "cum_prev_success_mean",
        "cum_prev_passive_rating_mean", "run_prev_rating_mean"
    ] if c in df.columns]

    # Select top coverage/variance neural cols for CV to avoid overfitting.
    def top_features(cols):
        rows = []
        for c in cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            rows.append({"feature": c, "coverage": vals.notna().mean(), "variance": vals.var(ddof=0)})
        q = pd.DataFrame(rows)
        if len(q) == 0:
            return []
        q = q[(q["coverage"] >= 0.1) & (q["variance"] > 0)]
        q = q.sort_values(["coverage", "variance"], ascending=False)
        return list(q.head(max_neural)["feature"])

    current_top = top_features(current_cols)
    prev_top = top_features(prev_cols)
    domain_top = top_features(domain_cols)

    sets = {
        "M0_condition_only": [c for c in ["is_up", "up_first", "global_trial_z_subject"] if c in df.columns],
        "M1_behavior_state": behavior_state,
        "M2_behavior_history": behavior_history,
    }
    if prev_top:
        sets["M3_behavior_plus_previous_neural_prospective"] = behavior_history + prev_top
    if current_top:
        sets["M4_behavior_plus_current_neural_explanatory"] = behavior_history + current_top
    if domain_top:
        sets["M5_behavior_plus_subject_domains"] = behavior_history + domain_top
    if prev_top and domain_top:
        sets["M6_behavior_plus_prev_neural_plus_domains"] = behavior_history + prev_top + domain_top
    return sets


def cv_ridge(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv_scheme: str = "subject_groupkfold", n_splits: int = 5) -> Tuple[Dict, Optional[pd.Series], Optional[List[str]], Optional[np.ndarray]]:
    if not HAS_SKLEARN:
        return {"model_ok": 0, "error": "sklearn unavailable"}, None, None, None

    ok = y.notna()
    X = X.loc[ok].copy()
    y = y.loc[ok].astype(float)
    groups = groups.loc[ok]

    # Numeric usable features.
    keep = []
    for c in X.columns:
        vals = pd.to_numeric(X[c], errors="coerce")
        if vals.notna().sum() >= 20 and vals.nunique(dropna=True) > 1:
            X[c] = vals
            keep.append(c)
    X = X[keep]
    if len(y) < 30 or X.shape[1] == 0:
        return {"model_ok": 0, "error": "too few rows or no features"}, None, keep, None

    if cv_scheme == "subject_groupkfold":
        n_groups = groups.nunique()
        splits = min(n_splits, n_groups)
        cv = GroupKFold(n_splits=splits)
        split_iter = cv.split(X, y, groups)
    else:
        splits = min(n_splits, len(y))
        cv = KFold(n_splits=splits, shuffle=True, random_state=42)
        split_iter = cv.split(X, y)

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 31)))
    ])

    preds = pd.Series(np.nan, index=X.index)
    fold_rows = []
    for fold, (tr, te) in enumerate(split_iter, 1):
        try:
            model.fit(X.iloc[tr], y.iloc[tr])
            pr = model.predict(X.iloc[te])
            preds.iloc[te] = pr
            fold_rows.append({
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "r2": r2_score(y.iloc[te], pr) if len(te) > 1 else np.nan,
                "mae": mean_absolute_error(y.iloc[te], pr),
                "rmse": math.sqrt(mean_squared_error(y.iloc[te], pr)),
            })
        except Exception as e:
            fold_rows.append({"fold": fold, "error": str(e)})

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

    coefs = None
    try:
        model.fit(X, y)
        coefs = model.named_steps["ridge"].coef_
    except Exception:
        pass

    return metrics, preds, keep, coefs


def run_cv(df: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    imp_rows = []
    pred_rows = []

    outcomes = {
        "all_reg_success": df,
        "up_success_only": df[df["is_up"].astype(float) == 1],
        "down_success_only": df[df["is_down"].astype(float) == 1],
    }

    for outcome, d in outcomes.items():
        for fs_name, cols in feature_sets.items():
            cols = [c for c in cols if c in d.columns]
            if not cols:
                continue
            for cv_scheme in ["subject_groupkfold", "trial_kfold"]:
                metrics, preds, feat_cols, coefs = cv_ridge(d[cols], d["success"], d["subject"], cv_scheme=cv_scheme)
                metrics.update({"outcome": outcome, "feature_set": fs_name, "features": ",".join(cols)})
                rows.append(metrics)
                if preds is not None:
                    prediction_rows = d[["subject", "run", "trial_in_run", "condition", "success"]].copy()
                    prediction_rows["outcome"] = outcome
                    prediction_rows["feature_set"] = fs_name
                    prediction_rows["cv_scheme"] = cv_scheme
                    prediction_rows["pred_success"] = preds
                    pred_rows.append(prediction_rows)
                if feat_cols is not None and coefs is not None:
                    for c, coef in zip(feat_cols, coefs):
                        imp_rows.append({
                            "outcome": outcome,
                            "feature_set": fs_name,
                            "cv_scheme": cv_scheme,
                            "feature": c,
                            "coef": float(coef),
                            "abs_coef": float(abs(coef))
                        })

    cv = pd.DataFrame(rows)
    imp = pd.DataFrame(imp_rows).sort_values("abs_coef", ascending=False) if imp_rows else pd.DataFrame()
    pred = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return cv, imp, pred


# ---------------------------
# Figures/report
# ---------------------------

def make_figures(outdir: Path, cv: pd.DataFrame, mech: pd.DataFrame, imp: pd.DataFrame) -> None:
    if not HAS_MPL:
        return
    figdir = outdir / "figures"
    mkdirp(figdir)

    if len(cv):
        plot = cv[(cv["cv_scheme"] == "subject_groupkfold") & (cv["model_ok"] == 1)].copy()
        if len(plot):
            plot = plot.sort_values(["outcome", "r2_cv_overall"])
            labels = plot["outcome"] + " | " + plot["feature_set"]
            plt.figure(figsize=(10, max(5, 0.35 * len(plot))))
            plt.barh(range(len(plot)), plot["r2_cv_overall"])
            plt.yticks(range(len(plot)), labels, fontsize=8)
            plt.xlabel("Cross-validated R² | subject GroupKFold")
            plt.title("Prediction of regulation success from behavior/neural state")
            plt.tight_layout()
            plt.savefig(figdir / "fig_01_prediction_cv_r2.png", dpi=200)
            plt.close()

    if len(mech):
        terms = ["is_up_c", "expected_pain_c", "is_up_c:expected_pain_c", "prev_rating_z_c", "prev_reg_success_c"]
        plot = mech[(mech["model_name"] == "behavior_dynamic_subject_FE") & (mech["term"].isin(terms))].copy()
        if len(plot):
            plt.figure(figsize=(8, 4))
            plt.errorbar(plot["term"], plot["beta"], yerr=1.96 * plot["se"], fmt="o")
            plt.axhline(0, linestyle="--", linewidth=1)
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("Beta ± 95% CI")
            plt.title("Behavioral dynamic state-dependence")
            plt.tight_layout()
            plt.savefig(figdir / "fig_02_behavior_state_dependence.png", dpi=200)
            plt.close()

    if len(imp):
        plot = imp[imp["cv_scheme"] == "subject_groupkfold"].head(25).copy()
        if len(plot):
            plt.figure(figsize=(9, max(5, 0.3 * len(plot))))
            plt.barh(range(len(plot)), plot["abs_coef"])
            plt.yticks(range(len(plot)), plot["feature"], fontsize=7)
            plt.xlabel("|ridge coefficient|")
            plt.title("Top prediction features")
            plt.tight_layout()
            plt.savefig(figdir / "fig_03_top_prediction_features.png", dpi=200)
            plt.close()


def write_report(outdir: Path, manifest: Dict, info: Dict, cv: pd.DataFrame, mech: pd.DataFrame, up: pd.DataFrame, down: pd.DataFrame, inventory: pd.DataFrame, neural_qc: pd.DataFrame) -> None:
    lines = []
    lines.append("# Trialwise neural-state integration for pain regulation - analysis ")
    lines.append("")
    lines.append("## Aim")
    lines.append("This analysis integrates behavioral/computational state with available trial-wise neural features to test mechanisms and prediction of up/down pain-regulation success.")
    lines.append("")
    lines.append("")
    lines.append("## Inputs and dimensions")
    lines.append(f"- Regulation trials: **{manifest.get('n_reg_trials')}**")
    lines.append(f"- Subjects: **{manifest.get('n_subjects')}**")
    lines.append(f"- Current-trial neural features: **{manifest.get('n_current_neural_features')}**")
    lines.append(f"- Previous-trial neural features: **{manifest.get('n_previous_neural_features')}**")
    lines.append(f"- Subject-level domain features: **{manifest.get('n_domain_features')}**")
    lines.append("")
    lines.append("## Neural file inventory")
    if len(inventory):
        show = inventory[["file", "status", "n_rows_raw", "n_cols_raw", "n_features", "merge_keys"]].head(30)
        lines.append(show.to_markdown(index=False))
    else:
        lines.append("No candidate neural files were found.")
    lines.append("")
    lines.append("## Main behavior/state-dependence baseline")
    if len(mech):
        show = mech[(mech["model_name"] == "behavior_dynamic_subject_FE") & (~mech["term"].astype(str).str.startswith("C(subject)"))].copy()
        lines.append(show[["term", "beta", "se", "t_or_z", "p", "q_fdr_target_terms", "n", "r2"]].to_markdown(index=False))
    else:
        lines.append("No mechanism tests available.")
    lines.append("")
    lines.append("## Neural mechanism interaction tests")
    if len(mech):
        show = mech[
            (mech["feature_type"] != "behavior") &
            (~mech["term"].astype(str).str.startswith("C(subject)")) &
            (mech["term"].astype(str) != "Intercept") &
            (mech["p"].notna())
        ].sort_values("p").head(30)
        if len(show):
            lines.append(show[["feature_type", "model_name", "term", "beta", "t_or_z", "p", "q_fdr_target_terms", "n", "r2"]].to_markdown(index=False))
        else:
            lines.append("No neural mechanism terms were estimable.")
    lines.append("")
    lines.append("## Up-specific neural/state models")
    if len(up):
        show = up[(~up["term"].astype(str).str.startswith("C(subject)")) & (up["p"].notna()) & (up["term"].astype(str) != "Intercept")].sort_values("p").head(25)
        lines.append(show[["feature_tested", "term", "beta", "t_or_z", "p", "q_fdr_target_terms", "n", "r2"]].to_markdown(index=False))
    else:
        lines.append("No up-specific models available.")
    lines.append("")
    lines.append("## Down-specific neural/state models")
    if len(down):
        show = down[(~down["term"].astype(str).str.startswith("C(subject)")) & (down["p"].notna()) & (down["term"].astype(str) != "Intercept")].sort_values("p").head(25)
        lines.append(show[["feature_tested", "term", "beta", "t_or_z", "p", "q_fdr_target_terms", "n", "r2"]].to_markdown(index=False))
    else:
        lines.append("No down-specific models available.")
    lines.append("")
    lines.append("## Cross-validated prediction")
    if len(cv):
        show = cv.sort_values(["outcome", "cv_scheme", "r2_cv_overall"], ascending=[True, True, False])
        lines.append(show[["outcome", "feature_set", "cv_scheme", "n", "n_features", "r2_cv_overall", "mae_cv_overall", "rmse_cv_overall"]].to_markdown(index=False))
    else:
        lines.append("No CV models available.")
    lines.append("")
    lines.append("## Interpretation guardrails")
    lines.append("- Previous-trial neural models are closer to prospective prediction.")
    lines.append("- Current-trial neural models are concurrent/explanatory and do not constitute pre-trial prediction.")
    lines.append("- A reliable improvement of behavior+previous-neural over behavior-only supports neural-state prediction.")
    lines.append("- A condition × neural feature interaction supports different up/down neural mechanisms.")
    lines.append("- A condition × expected pain × neural feature interaction supports neural modulation of state-dependence.")
    lines.append("")
    (outdir / "trialwise_neural_state_integration_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derivatives", required=True)
    ap.add_argument("--trialwise", default=None)
    ap.add_argument("--domain-scores", default=None)
    ap.add_argument("--neural-table", action="append", default=[], help="Optional explicit trialwise neural table path. Can be repeated.")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max-features-per-file", type=int, default=1000)
    ap.add_argument("--max-lag-neural-features", type=int, default=500)
    ap.add_argument("--top-n-tests", type=int, default=40)
    ap.add_argument("--no-auto-search", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    derivatives = Path(args.derivatives)
    outdir = Path(args.outdir) if args.outdir else derivatives / "trialwise_neural_state_integration_analysis"
    mkdirp(outdir)
    mkdirp(outdir / "tables")
    mkdirp(outdir / "figures")

    trialwise_path = Path(args.trialwise) if args.trialwise else infer_trialwise_path(derivatives)
    if trialwise_path is None or not trialwise_path.exists():
        raise FileNotFoundError("Could not find trialwise computational table. Provide --trialwise.")

    domain_path = Path(args.domain_scores) if args.domain_scores else infer_domain_scores_path(derivatives)

    trialwise = safe_read_tsv(trialwise_path)
    if trialwise is None:
        raise RuntimeError(f"Could not read {trialwise_path}")

    reg, info = make_behavior_state_table(trialwise)

    neural_paths = [Path(p) for p in args.neural_table if Path(p).exists()]
    if not args.no_auto_search:
        for p in candidate_neural_files(derivatives):
            if p not in neural_paths:
                neural_paths.append(p)

    reg2, inventory, neural_cols_raw = merge_neural_tables(reg, neural_paths, max_features_per_file=args.max_features_per_file)
    reg3, current_and_prev_cols, neural_qc = create_neural_lag_features(reg2, neural_cols_raw, max_lag_cols=args.max_lag_neural_features)
    current_cols = [c for c in current_and_prev_cols if c.endswith("_current_z")]
    prev_cols = [c for c in current_and_prev_cols if c.endswith("_prev_z")]

    reg4, domain_cols = attach_domain_scores(reg3, domain_path)

    # Save integrated table and QC.
    reg4.to_csv(outdir / "tables" / "trialwise_neural_state_table.tsv", sep="\t", index=False)
    inventory.to_csv(outdir / "tables" / "neural_file_inventory.tsv", sep="\t", index=False)
    neural_qc.to_csv(outdir / "tables" / "neural_feature_qc.tsv", sep="\t", index=False)

    # Mechanism models.
    mech = run_neural_mechanism_tests(reg4, current_cols, prev_cols, domain_cols, top_n=args.top_n_tests)
    mech.to_csv(outdir / "tables" / "neural_mechanism_interaction_tests.tsv", sep="\t", index=False)

    up, down = run_condition_specific_neural_models(reg4, current_cols, prev_cols, top_n=args.top_n_tests)
    up.to_csv(outdir / "tables" / "up_neural_success_models.tsv", sep="\t", index=False)
    down.to_csv(outdir / "tables" / "down_neural_success_models.tsv", sep="\t", index=False)

    # CV prediction.
    feature_sets = make_feature_sets(reg4, current_cols, prev_cols, domain_cols)
    cv, imp, preds = run_cv(reg4, feature_sets)
    cv.to_csv(outdir / "tables" / "prospective_neural_prediction_cv.tsv", sep="\t", index=False)
    imp.to_csv(outdir / "tables" / "feature_importance_ridge_neural.tsv", sep="\t", index=False)
    preds.to_csv(outdir / "tables" / "cv_neural_predictions.tsv", sep="\t", index=False)

    if not args.no_figures:
        make_figures(outdir, cv, mech, imp)

    manifest = {
        "derivatives": str(derivatives),
        "trialwise": str(trialwise_path),
        "domain_scores": str(domain_path) if domain_path else None,
        "outdir": str(outdir),
        "n_reg_trials": int(len(reg4)),
        "n_subjects": int(reg4["subject"].nunique()),
        "n_candidate_neural_files": int(len(neural_paths)),
        "n_usable_neural_files": int((inventory["status"] == "usable").sum()) if len(inventory) else 0,
        "n_raw_neural_features": int(len(neural_cols_raw)),
        "n_current_neural_features": int(len(current_cols)),
        "n_previous_neural_features": int(len(prev_cols)),
        "n_domain_features": int(len(domain_cols)),
        "success_col": info.get("success_col"),
        "expected_col": info.get("expected_col"),
        "has_statsmodels": HAS_STATSMODELS,
        "has_sklearn": HAS_SKLEARN,
        "has_scipy": HAS_SCIPY,
        "interpretation": {
            "previous_neural": "closer to prospective state prediction",
            "current_neural": "same-trial explanatory neural mechanism, not pre-trial prediction; behavioral/design leakage columns excluded",
            "domain_features": "subject-level individual-difference neural domains"
        }
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    write_report(outdir, manifest, info, cv, mech, up, down, inventory, neural_qc)

    print(f"Processing completed: {outdir}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
