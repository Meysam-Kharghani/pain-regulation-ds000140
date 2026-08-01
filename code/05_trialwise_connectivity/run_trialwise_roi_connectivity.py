#!/usr/bin/env python3
"""
run_trialwise_roi_connectivity.py

Trialwise ROI LSS + beta-series FC + directed lagged beta-series EC proxy
for ds000140 self-regulation pain project.

Purpose
-------
This script estimates trialwise activity for all selected ROIs and then tests:
  1) condition-wise ROI activity,
  2) beta-series functional connectivity (FC),
  3) directed lagged beta-series predictive connectivity
     as an effective-connectivity-like exploratory proxy (EC-lite).

Important EC note
-----------------
The EC output is not a DCM result and should not be described as definitive
causal effective connectivity. It is a directed, lagged predictive model:

  target_beta[t+1] ~ source_beta[t] + target_beta[t] + temperature[t+1] + run covariates

The coefficient source_beta[t] -> target_beta[t+1] is a directed predictive
coupling estimate. It is useful as an exploratory EC proxy and for prioritizing
pathways before model-based EC analyses such as DCM, spectral DCM, SEM, or
state-space models.

Inputs
------
--inventory:
  first_level_run_inventory.tsv from first_level_inputs

--roi-definitions:
  roi_masks/roi_definitions.tsv

--behavior-trials:
  optional behavior_trials_master_clean.tsv

Outputs
-------
outdir/
  manifest.json
  trialwise_roi_fc_ec_report.md
  tables/
    trialwise_roi_lss_long.tsv
    trialwise_roi_lss_wide.tsv
    roi_activity_subject_summary.tsv
    roi_activity_subject_tests.tsv
    fc_beta_series_edges.tsv
    fc_edge_contrast_tests.tsv
    ec_lagged_edges.tsv
    ec_lagged_edge_contrast_tests.tsv
    pathway_edge_inventory.tsv
    run_inventory_roi_lss.tsv
  per_run/
    sub-XX_run-YY_roi_lss.tsv

Example
-------
python run_trialwise_roi_connectivity.py \
  --inventory <PROJECT_ROOT>/derivatives/first_level_inputs/tables/first_level_run_inventory.tsv \
  --roi-definitions <PROJECT_ROOT>/derivatives/roi_masks/roi_definitions.tsv \
  --behavior-trials <PROJECT_ROOT>/derivatives/beh/behavior_trials_master_clean.tsv \
  --outdir <PROJECT_ROOT>/derivatives/trialwise_roi_connectivity \
  --bold-column bold_unsmoothed \
  --include-rating-nuisance \
  --rating-events-column events_long \
  --resume

For a small validation run:
python run_trialwise_roi_connectivity.py ... --max-runs 3 --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
import traceback
from datetime import datetime
from itertools import combinations, permutations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from nilearn.image import resample_to_img
from nilearn.masking import apply_mask
from nilearn.glm.first_level import make_first_level_design_matrix


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_tsv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=False)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def clean_subject(x) -> str:
    s = str(x)
    if s.startswith("sub-"):
        return s
    m = re.search(r"(\d+)", s)
    return f"sub-{int(m.group(1)):02d}" if m else s


def sanitize_name(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    if not x:
        x = "roi"
    return x


def infer_path(row: pd.Series, candidates: Sequence[str], label: str, preferred: Optional[str] = None) -> Path:
    cols = []
    if preferred:
        cols.append(preferred)
    cols.extend([c for c in candidates if c != preferred])

    for c in cols:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            p = Path(str(row[c])).expanduser()
            if p.exists():
                return p.resolve()

    for c in cols:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            p = Path(str(row[c])).expanduser()
            raise FileNotFoundError(f"{label} path from column {c} does not exist: {p}")

    raise KeyError(f"Could not infer {label}. Tried columns: {cols}")


def infer_optional_path(
    row: pd.Series,
    candidates: Sequence[str],
    preferred: Optional[str] = None,
) -> Optional[Path]:
    """Return the first existing path from the requested inventory columns."""
    cols: List[str] = []
    if preferred:
        cols.append(preferred)
    cols.extend([c for c in candidates if c != preferred])
    for c in cols:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            candidate = Path(str(row[c])).expanduser()
            if candidate.exists():
                return candidate.resolve()
    return None


def get_subject_run(row: pd.Series) -> Tuple[str, int, str]:
    sub = None
    for c in ["subject", "sub", "participant_id"]:
        if c in row.index and pd.notna(row[c]):
            sub = clean_subject(row[c])
            break
    if sub is None:
        raise KeyError("No subject column found.")

    run = None
    for c in ["run", "run_id", "run_number"]:
        if c in row.index and pd.notna(row[c]):
            run = int(float(row[c]))
            break
    if run is None and "run_label" in row.index and pd.notna(row["run_label"]):
        m = re.search(r"(\d+)", str(row["run_label"]))
        if m:
            run = int(m.group(1))
    if run is None:
        raise KeyError("No run column found.")

    run_label = f"run-{run:02d}"
    if "run_label" in row.index and pd.notna(row["run_label"]):
        run_label = str(row["run_label"])
    return sub, run, run_label


def get_tr(row: pd.Series, img: nib.spatialimages.SpatialImage) -> float:
    for c in ["tr", "TR", "repetition_time"]:
        if c in row.index and pd.notna(row[c]):
            val = float(row[c])
            if val > 0:
                return val
    zooms = img.header.get_zooms()
    if len(zooms) >= 4 and zooms[3] > 0:
        return float(zooms[3])
    raise ValueError("Could not determine TR.")


def one_sample_test(values, label: str) -> Dict[str, object]:
    x = safe_num(pd.Series(values)).dropna().to_numpy(float)
    out = {"quantity": label, "n": int(len(x))}
    if len(x) < 2:
        out.update({"mean": np.nan, "sd": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "dz": np.nan})
        return out
    tt = stats.ttest_1samp(x, 0.0)
    sd = x.std(ddof=1)
    out.update({
        "mean": float(x.mean()),
        "sd": float(sd),
        "se": float(sd / np.sqrt(len(x))),
        "t": float(tt.statistic),
        "p": float(tt.pvalue),
        "dz": float(x.mean() / sd) if sd > 1e-12 else np.nan,
    })
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(x).pvalue)
    except Exception:
        out["wilcoxon_p"] = np.nan
    return out


def fdr(df: pd.DataFrame, p_col: str, q_col: str, by: Sequence[str] = ()) -> pd.DataFrame:
    out = df.copy()
    out[q_col] = np.nan
    if out.empty or p_col not in out.columns:
        return out

    groups = out.groupby(list(by), dropna=False).groups.values() if by else [out.index]
    for idx in groups:
        p = pd.to_numeric(out.loc[idx, p_col], errors="coerce")
        mask = p.notna() & np.isfinite(p)
        if mask.sum() > 0:
            out.loc[p[mask].index, q_col] = multipletests(p[mask], method="fdr_bh")[1]
    return out


def fisher_z(r: float) -> float:
    if pd.isna(r) or not np.isfinite(r):
        return np.nan
    r = max(min(float(r), 0.999999), -0.999999)
    return float(np.arctanh(r))


def safe_corr(x, y) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 4:
        return np.nan
    if d["x"].std(ddof=1) < 1e-12 or d["y"].std(ddof=1) < 1e-12:
        return np.nan
    return float(np.corrcoef(d["x"], d["y"])[0, 1])


# =============================================================================
# ROI definitions
# =============================================================================

def find_col(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def load_roi_definitions(path: Path, roi_regex: str = "", exclude_regex: str = "") -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = df.columns.tolist()

    name_col = find_col(cols, ["roi", "roi_name", "name", "label", "mask_name"])
    mask_col = find_col(cols, ["mask", "mask_path", "path", "file", "resampled_map"])
    status_col = find_col(cols, ["status", "mask_status"])

    if name_col is None:
        raise ValueError(f"Could not identify ROI name column in {path}. Columns={cols}")
    if mask_col is None:
        raise ValueError(f"Could not identify ROI mask path column in {path}. Columns={cols}")

    out = df.copy()
    out = out.rename(columns={name_col: "roi", mask_col: "mask_path"})
    if status_col is not None and status_col in out.columns:
        status = out[status_col].astype(str).str.upper()
        # Retain rows marked OK, without excluding rows when an alternative status vocabulary is used.
        if (status == "OK").any():
            out = out[status == "OK"].copy()

    out["roi"] = out["roi"].astype(str)
    out["roi_safe"] = out["roi"].map(sanitize_name)

    # Resolve paths relative to roi_definitions.tsv directory if needed.
    base = path.parent
    resolved = []
    for p in out["mask_path"].astype(str):
        pp = Path(p).expanduser()
        if not pp.is_absolute():
            pp = base / pp
        resolved.append(str(pp.resolve()))
    out["mask_path"] = resolved

    out = out[out["mask_path"].map(lambda p: Path(p).exists())].copy()

    if roi_regex:
        pat = re.compile(roi_regex, flags=re.IGNORECASE)
        out = out[out["roi"].map(lambda x: bool(pat.search(str(x))))].copy()
    if exclude_regex:
        pat = re.compile(exclude_regex, flags=re.IGNORECASE)
        out = out[~out["roi"].map(lambda x: bool(pat.search(str(x))))].copy()

    out = out.drop_duplicates("roi_safe").reset_index(drop=True)
    if out.empty:
        raise ValueError("No ROI masks available after filtering.")

    return out[["roi", "roi_safe", "mask_path"] + [c for c in out.columns if c not in ["roi", "roi_safe", "mask_path"]]]


def load_or_resample_mask(mask_path: Path, ref_img: nib.spatialimages.SpatialImage, cache_dir: Path) -> Path:
    mask_img = nib.load(str(mask_path))
    if mask_img.shape[:3] == ref_img.shape[:3] and np.allclose(mask_img.affine, ref_img.affine, atol=1e-4):
        return mask_path

    ensure_dir(cache_dir)
    out = cache_dir / (mask_path.name.replace(".nii.gz", "").replace(".nii", "") + "_resampled_to_bold.nii.gz")
    if out.exists() and out.stat().st_size > 0:
        return out

    res = resample_to_img(mask_img, ref_img, interpolation="nearest", force_resample=True, copy_header=True)
    data = np.asanyarray(res.dataobj)
    bin_data = (np.nan_to_num(data, nan=0.0) > 0).astype(np.uint8)
    nib.Nifti1Image(bin_data, ref_img.affine, ref_img.header.copy()).to_filename(str(out))
    return out


# =============================================================================
# Events and confounds
# =============================================================================

def read_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "onset" not in df.columns:
        raise ValueError(f"Events lacks onset column: {path}")
    if "duration" not in df.columns:
        df["duration"] = 0.0

    df = df.copy()
    df["onset"] = safe_num(df["onset"])
    df["duration"] = safe_num(df["duration"]).fillna(0.0)

    if "condition" in df.columns:
        df["condition"] = df["condition"].astype(str).str.lower().str.strip()
    elif "trial_type" in df.columns:
        tt = df["trial_type"].astype(str).str.lower()
        df["condition"] = np.where(tt.str.contains("up"), "up",
                           np.where(tt.str.contains("down"), "down",
                           np.where(tt.str.contains("passive|stim"), "passive", "")))
    else:
        df["condition"] = "stim"

    df = df[df["condition"].isin(["passive", "up", "down"])].copy()
    df = df.sort_values("onset").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No passive/up/down stimulation events found: {path}")

    df["trial_lss_index"] = np.arange(1, len(df) + 1, dtype=int)

    temp_candidates = ["temperature_z_subject", "temperature_z", "temp_z", "temperature", "temp", "stimulus_intensity", "intensity", "modulation"]
    rating_candidates = ["rating_z_subject", "rating_z", "rating", "rating_raw", "pain_rating", "pain"]

    temp_col = next((c for c in temp_candidates if c in df.columns), None)
    rating_col = next((c for c in rating_candidates if c in df.columns), None)

    df["temperature_for_model"] = safe_num(df[temp_col]) if temp_col else np.nan
    df["temperature_source_col"] = temp_col or ""
    df["rating_for_model"] = safe_num(df[rating_col]) if rating_col else np.nan
    df["rating_source_col"] = rating_col or ""

    return df


def read_rating_events(path: Optional[Path]) -> pd.DataFrame:
    """Read rating-period events from an events_long-style TSV."""
    columns = ["onset", "duration", "trial_type", "modulation"]
    if path is None or not path.exists():
        return pd.DataFrame(columns=columns)

    raw = pd.read_csv(path, sep="\t")
    if "onset" not in raw.columns:
        return pd.DataFrame(columns=columns)
    if "duration" not in raw.columns:
        raw["duration"] = 0.0

    phase = (
        raw["event_phase"].astype(str).str.lower()
        if "event_phase" in raw.columns
        else pd.Series("", index=raw.index)
    )
    trial_type = (
        raw["trial_type"].astype(str).str.lower()
        if "trial_type" in raw.columns
        else pd.Series("", index=raw.index)
    )
    mask = phase.eq("rating") | trial_type.str.contains("rating|response", regex=True, na=False)
    ratings = raw.loc[mask, ["onset", "duration"]].copy()
    ratings["onset"] = safe_num(ratings["onset"])
    ratings["duration"] = safe_num(ratings["duration"]).fillna(0.0)
    ratings = ratings.dropna(subset=["onset"])
    ratings = ratings[ratings["duration"] >= 0].copy()
    ratings["trial_type"] = "rating_period"
    ratings["modulation"] = 1.0
    return ratings[columns].sort_values("onset").reset_index(drop=True)


def rating_events_from_trial_table(events: pd.DataFrame) -> pd.DataFrame:
    """Construct rating-period events from canonical behavior timing columns."""
    columns = ["onset", "duration", "trial_type", "modulation"]
    onset_col = next(
        (c for c in ["rating_onset_runrel", "rating_onset", "rating_onset_abs"] if c in events.columns),
        None,
    )
    duration_col = next(
        (c for c in ["rating_duration", "response_duration"] if c in events.columns),
        None,
    )
    if onset_col is None:
        return pd.DataFrame(columns=columns)

    ratings = pd.DataFrame({
        "onset": safe_num(events[onset_col]),
        "duration": safe_num(events[duration_col]).fillna(0.0) if duration_col else 0.0,
    })
    ratings = ratings.dropna(subset=["onset"])
    ratings = ratings[ratings["duration"] >= 0].copy()
    ratings["trial_type"] = "rating_period"
    ratings["modulation"] = 1.0
    return ratings[columns].sort_values("onset").reset_index(drop=True)


def maybe_merge_behavior(events: pd.DataFrame, behavior: Optional[pd.DataFrame], subject: str, run: int) -> pd.DataFrame:
    if behavior is None:
        return events

    bh = behavior.copy()
    if "subject" not in bh.columns or "run" not in bh.columns:
        return events

    bh["subject"] = bh["subject"].map(clean_subject)
    bh["run"] = safe_num(bh["run"]).astype("Int64")
    bh = bh[(bh["subject"] == subject) & (bh["run"] == run)].copy()
    if bh.empty:
        return events

    ev = events.copy()
    trial_cols = ["trial_in_run", "trial", "trial_index", "trial_lss_index"]
    b_trial = next((c for c in trial_cols if c in bh.columns), None)
    if b_trial is None:
        return ev

    bh["_trial_key"] = safe_num(bh[b_trial]).astype("Int64")
    ev["_trial_key"] = ev["trial_lss_index"].astype("Int64")

    keep_cols = ["_trial_key"]
    for c in bh.columns:
        if c not in ["subject", "run", "_trial_key"] and c not in ev.columns:
            keep_cols.append(c)

    merged = ev.merge(bh[keep_cols], on="_trial_key", how="left")
    merged = merged.drop(columns=["_trial_key"], errors="ignore")

    for tc in ["temperature_z_subject", "temperature_z", "temperature", "temp"]:
        if tc in merged.columns and merged["temperature_for_model"].isna().mean() > 0.5:
            merged["temperature_for_model"] = safe_num(merged[tc])
            merged["temperature_source_col"] = tc
            break

    for rc in ["rating_z_subject", "rating_z", "rating_raw", "rating", "pain_rating"]:
        if rc in merged.columns and merged["rating_for_model"].isna().mean() > 0.5:
            merged["rating_for_model"] = safe_num(merged[rc])
            merged["rating_source_col"] = rc
            break

    return merged


def read_confounds(path: Optional[Path], n_vols: int) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(index=np.arange(n_vols))

    df = pd.read_csv(path, sep="\t")
    out = pd.DataFrame(index=np.arange(len(df)))
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() > 0:
            out[c] = vals

    if len(out) < n_vols:
        pad = pd.DataFrame(0.0, index=np.arange(n_vols - len(out)), columns=out.columns)
        out = pd.concat([out, pad], ignore_index=True)
    elif len(out) > n_vols:
        out = out.iloc[:n_vols].copy()

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    keep = [c for c in out.columns if out[c].std() > 1e-12 or out[c].abs().sum() > 0]
    out = out[keep].copy()

    new_cols = []
    seen = {}
    for c in out.columns:
        nc = sanitize_name(c)
        if nc in seen:
            seen[nc] += 1
            nc = f"{nc}_{seen[nc]}"
        else:
            seen[nc] = 0
        new_cols.append(nc)
    out.columns = new_cols
    return out


# =============================================================================
# Trialwise LSS
# =============================================================================

def extract_roi_timeseries(bold_path: Path, roi_defs: pd.DataFrame, cache_dir: Path) -> Tuple[np.ndarray, List[str], List[str], Dict[str, object]]:
    img = nib.load(str(bold_path))
    ts_list = []
    names = []
    original_names = []
    voxel_counts = {}

    for _, r in roi_defs.iterrows():
        roi = r["roi"]
        roi_safe = r["roi_safe"]
        mask_path = Path(r["mask_path"])
        try:
            mpath = load_or_resample_mask(mask_path, img, cache_dir=cache_dir)
            mask_img = nib.load(str(mpath))
            n_vox = int(np.sum(np.asanyarray(mask_img.dataobj) > 0))
            if n_vox < 1:
                continue
            dat = apply_mask(str(bold_path), str(mpath)).astype(np.float64)
            if dat.ndim == 1:
                dat = dat[:, None]
            dat = np.nan_to_num(dat, nan=0.0, posinf=0.0, neginf=0.0)
            y = dat.mean(axis=1)
            if np.std(y) < 1e-12:
                continue
            ts_list.append(y)
            names.append(roi_safe)
            original_names.append(roi)
            voxel_counts[roi_safe] = n_vox
        except Exception:
            continue

    if not ts_list:
        raise ValueError("No valid ROI time series extracted.")

    Y = np.column_stack(ts_list)
    meta = {
        "n_rois_extracted": int(Y.shape[1]),
        "roi_names": names,
        "roi_original_names": original_names,
        "roi_voxel_counts": voxel_counts,
    }
    return Y, names, original_names, meta


def build_lss_events(
    events: pd.DataFrame,
    target_idx: int,
    other_mode: str,
    rating_events: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build one LSS design with an optional pooled rating-period nuisance."""
    target = events.iloc[target_idx]
    rows = [{
        "onset": float(target["onset"]),
        "duration": float(target["duration"]),
        "trial_type": "target_trial",
        "modulation": 1.0,
    }]

    others = events.drop(events.index[target_idx]).copy()
    if other_mode == "by_condition":
        for _, r in others.iterrows():
            rows.append({
                "onset": float(r["onset"]),
                "duration": float(r["duration"]),
                "trial_type": f"other_{r['condition']}",
                "modulation": 1.0,
            })
    else:
        for _, r in others.iterrows():
            rows.append({
                "onset": float(r["onset"]),
                "duration": float(r["duration"]),
                "trial_type": "other_stim",
                "modulation": 1.0,
            })

    if rating_events is not None and not rating_events.empty:
        rows.extend(rating_events.to_dict(orient="records"))

    return pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)


def fit_lss_for_run(row: pd.Series,
                    roi_defs: pd.DataFrame,
                    behavior: Optional[pd.DataFrame],
                    out_run_path: Path,
                    cache_dir: Path,
                    bold_column: str,
                    hrf_model: str,
                    drift_model: str,
                    high_pass: float,
                    other_mode: str,
                    include_rating_nuisance: bool,
                    rating_events_column: str,
                    solver_rcond: float,
                    target_unique_tol: float) -> Dict[str, object]:
    subject, run, run_label = get_subject_run(row)

    bold = infer_path(
        row,
        ["bold_unsmoothed", "bold", "bold_smoothed", "smoothed_bold", "bold_file", "bold_path", "preproc_bold", "func"],
        "BOLD",
        preferred=bold_column if bold_column else None,
    )
    events_path = infer_path(
        row,
        ["events_stim", "events", "events_file", "events_path", "events_long"],
        "events",
    )
    rating_events_path = infer_optional_path(
        row,
        ["events_long", "events_all", "events_with_ratings", "events"],
        preferred=rating_events_column if rating_events_column else None,
    )

    conf_path = None
    for c in ["confounds_glm", "confounds", "confounds_file", "confounds_path"]:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            p = Path(str(row[c])).expanduser()
            if p.exists():
                conf_path = p.resolve()
                break

    img = nib.load(str(bold))
    n_vols = int(img.shape[-1])
    tr = get_tr(row, img)
    frame_times = np.arange(n_vols, dtype=float) * tr

    events = read_events(events_path)
    events = maybe_merge_behavior(events, behavior, subject, run)

    rating_events = pd.DataFrame(
        columns=["onset", "duration", "trial_type", "modulation"]
    )
    rating_event_source = "not_requested"
    if include_rating_nuisance:
        rating_events = read_rating_events(rating_events_path)
        rating_event_source = str(rating_events_path) if not rating_events.empty else ""
        if rating_events.empty:
            rating_events = rating_events_from_trial_table(events)
            if not rating_events.empty:
                rating_event_source = "behavior_timing_columns"
        if rating_events.empty:
            raise ValueError(
                "Rating nuisance was requested, but no rating events could be "
                "recovered from events_long or behavior timing columns."
            )

    confounds = read_confounds(conf_path, n_vols)

    Y, roi_names, roi_original_names, roi_meta = extract_roi_timeseries(
        bold_path=bold,
        roi_defs=roi_defs,
        cache_dir=cache_dir / subject / f"run-{run:02d}",
    )
    if Y.shape[0] != n_vols:
        raise ValueError(f"ROI time series length {Y.shape[0]} != n_vols {n_vols}")

    rows = []
    for i in range(len(events)):
        lss_events = build_lss_events(
            events,
            i,
            other_mode=other_mode,
            rating_events=rating_events if include_rating_nuisance else None,
        )

        try:
            design = make_first_level_design_matrix(
                frame_times=frame_times,
                events=lss_events,
                hrf_model=hrf_model,
                drift_model=drift_model,
                high_pass=high_pass,
                add_regs=confounds if confounds.shape[1] > 0 else None,
                add_reg_names=list(confounds.columns) if confounds.shape[1] > 0 else None,
            )
        except TypeError:
            design = make_first_level_design_matrix(
                frame_times,
                events=lss_events,
                hrf_model=hrf_model,
                drift_model=drift_model,
                high_pass=high_pass,
                add_regs=confounds if confounds.shape[1] > 0 else None,
                add_reg_names=list(confounds.columns) if confounds.shape[1] > 0 else None,
            )

        if "target_trial" not in design.columns:
            raise ValueError("Design lacks target_trial column.")

        X = np.asarray(design.values, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        target_col = list(design.columns).index("target_trial")

        # Full-pseudoinverse estimate retained only as a numerical audit.
        beta_full_pinv = np.linalg.pinv(X, rcond=solver_rcond)[target_col, :] @ Y

        # Frisch-Waugh-Lovell estimate of the target-trial coefficient.
        # Nuisance columns are L2-scaled before the SVD so that the numerical
        # cutoff is not dominated by heterogeneous column units. Scaling does
        # not change their column space. This estimate remains unique even if
        # nuisance coefficients themselves are not uniquely identifiable.
        target_vec = X[:, target_col]
        nuisance = np.delete(X, target_col, axis=1)
        nuisance_norms = np.linalg.norm(nuisance, axis=0)
        keep_nuisance = nuisance_norms > np.finfo(float).eps
        nuisance = nuisance[:, keep_nuisance]
        nuisance_norms = nuisance_norms[keep_nuisance]

        if nuisance.shape[1] > 0:
            nuisance_scaled = nuisance / nuisance_norms
            nuisance_pinv = np.linalg.pinv(nuisance_scaled, rcond=solver_rcond)
            target_resid = target_vec - nuisance_scaled @ (nuisance_pinv @ target_vec)
        else:
            nuisance_scaled = nuisance
            target_resid = target_vec.copy()

        target_norm = max(float(np.linalg.norm(target_vec)), np.finfo(float).eps)
        target_relative_residual = float(np.linalg.norm(target_resid) / target_norm)
        if target_relative_residual <= target_unique_tol:
            raise ValueError(
                "target_trial is not uniquely estimable: "
                f"relative residual={target_relative_residual:.6g}, "
                f"threshold={target_unique_tol:.6g}"
            )

        denominator = float(target_resid @ target_vec)
        if not np.isfinite(denominator) or abs(denominator) <= np.finfo(float).eps:
            raise ValueError(f"Invalid FWL denominator: {denominator!r}")
        beta = (target_resid @ Y) / denominator

        beta_delta = np.asarray(beta - beta_full_pinv, dtype=float)
        beta_scale = np.maximum(np.abs(beta_full_pinv), 1.0)
        max_abs_beta_delta = float(np.max(np.abs(beta_delta)))
        max_rel_beta_delta = float(np.max(np.abs(beta_delta) / beta_scale))
        target_vif = float(1.0 / max(target_relative_residual ** 2, np.finfo(float).eps))

        scaled_columns = []
        for j in range(X.shape[1]):
            col = X[:, j]
            norm = float(np.linalg.norm(col))
            if norm > np.finfo(float).eps:
                scaled_columns.append(col / norm)
        if scaled_columns:
            X_scaled = np.column_stack(scaled_columns)
            singular_values = np.linalg.svd(X_scaled, compute_uv=False)
            sv_keep = singular_values > solver_rcond * singular_values[0]
            scaled_condition_number = (
                float(singular_values[sv_keep].max() / singular_values[sv_keep].min())
                if np.any(sv_keep) else np.inf
            )
        else:
            scaled_condition_number = np.inf

        ev = events.iloc[i]
        base = {
            "subject": subject,
            "run": run,
            "run_label": run_label,
            "trial_lss_index": int(ev["trial_lss_index"]),
            "condition": ev["condition"],
            "onset": float(ev["onset"]),
            "duration": float(ev["duration"]),
            "temperature_for_model": float(ev["temperature_for_model"]) if pd.notna(ev["temperature_for_model"]) else np.nan,
            "temperature_source_col": ev.get("temperature_source_col", ""),
            "rating_for_model": float(ev["rating_for_model"]) if pd.notna(ev["rating_for_model"]) else np.nan,
            "rating_source_col": ev.get("rating_source_col", ""),
            "n_vols": n_vols,
            "tr": tr,
            "design_n_cols": int(X.shape[1]),
            "design_rank": int(np.linalg.matrix_rank(X)),
            "lss_solver": "FWL_target_residualization",
            "solver_rcond": float(solver_rcond),
            "target_unique_tol": float(target_unique_tol),
            "target_relative_residual": target_relative_residual,
            "target_vif": target_vif,
            "scaled_condition_number": scaled_condition_number,
            "max_abs_beta_delta_from_full_pinv": max_abs_beta_delta,
            "max_rel_beta_delta_from_full_pinv": max_rel_beta_delta,
            "hrf_model": hrf_model,
            "drift_model": drift_model,
            "high_pass": high_pass,
            "other_mode": other_mode,
            "include_rating_nuisance": bool(include_rating_nuisance),
            "n_rating_events": int(len(rating_events)) if include_rating_nuisance else 0,
            "rating_event_source": rating_event_source,
            "bold": str(bold),
            "events_file": str(events_path),
            "confounds_file": str(conf_path) if conf_path is not None else "",
        }
        for roi_safe, val in zip(roi_names, beta):
            base[f"roi_{roi_safe}"] = float(val)
        rows.append(base)

    out_df = pd.DataFrame(rows)
    save_tsv(out_df, out_run_path)

    return {
        "subject": subject,
        "run": run,
        "run_label": run_label,
        "status": "OK",
        "n_trials": int(len(out_df)),
        "n_rois_extracted": int(len(roi_names)),
        "out_file": str(out_run_path),
        "bold": str(bold),
        "events_file": str(events_path),
    }


# =============================================================================
# Long/wide conversion and behavior z-scoring
# =============================================================================

def wide_to_long(trial_wide: pd.DataFrame, roi_defs: pd.DataFrame) -> pd.DataFrame:
    roi_cols = [c for c in trial_wide.columns if c.startswith("roi_")]
    id_cols = [c for c in trial_wide.columns if c not in roi_cols]
    long = trial_wide.melt(id_vars=id_cols, value_vars=roi_cols, var_name="roi_safe_col", value_name="roi_beta")
    long["roi_safe"] = long["roi_safe_col"].str.replace("^roi_", "", regex=True)
    mapping = roi_defs[["roi", "roi_safe"]].drop_duplicates()
    long = long.merge(mapping, on="roi_safe", how="left")
    long = long.drop(columns=["roi_safe_col"])
    return long


def zscore_temperature_within_subject(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temperature_z_subject_lss"] = np.nan
    for sub, idx in out.groupby("subject").groups.items():
        vals = safe_num(out.loc[idx, "temperature_for_model"])
        if vals.notna().sum() >= 2 and vals.std(ddof=1) > 1e-12:
            out.loc[idx, "temperature_z_subject_lss"] = (vals - vals.mean()) / vals.std(ddof=1)
        else:
            out.loc[idx, "temperature_z_subject_lss"] = vals
    return out


# =============================================================================
# ROI activity analysis
# =============================================================================

def roi_activity_subject_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    g = long_df.groupby(["subject", "roi", "roi_safe", "condition"], as_index=False).agg(
        mean_beta=("roi_beta", "mean"),
        n_trials=("roi_beta", "size"),
    )
    wide = g.pivot(index=["subject", "roi", "roi_safe"], columns="condition", values="mean_beta").reset_index()
    wide.columns.name = None
    for c in ["passive", "up", "down"]:
        if c not in wide.columns:
            wide[c] = np.nan
    wide = wide.rename(columns={"passive": "beta_passive", "up": "beta_up", "down": "beta_down"})
    wide["up_minus_down"] = wide["beta_up"] - wide["beta_down"]
    wide["up_minus_passive"] = wide["beta_up"] - wide["beta_passive"]
    wide["down_minus_passive"] = wide["beta_down"] - wide["beta_passive"]
    wide["regulation_mean_minus_passive"] = 0.5 * (wide["beta_up"] + wide["beta_down"]) - wide["beta_passive"]
    return wide


def roi_activity_tests(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (roi, roi_safe), d in summary.groupby(["roi", "roi_safe"]):
        for q in ["beta_passive", "beta_up", "beta_down", "up_minus_down", "up_minus_passive", "down_minus_passive", "regulation_mean_minus_passive"]:
            if q not in d.columns:
                continue
            row = one_sample_test(d[q], q)
            row.update({"roi": roi, "roi_safe": roi_safe})
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = fdr(out, "p", "q_fdr_by_quantity", by=["quantity"])
    return out


# =============================================================================
# Pathway inventory
# =============================================================================

DEFAULT_PATHWAY_RULES = [
    ("NAc_vmPFC", r"(NAc|accumbens)", r"(vmPFC|mOFC|medial_orbitofrontal|orbitofrontal)"),
    ("pgACC_rACC_PAG", r"(pgACC|rACC|rostral|perigenual)", r"(PAG|periaqueductal)"),
    ("dlPFC_IFJ_vmPFC", r"(dlPFC|IFJ|inferior_frontal)", r"(vmPFC|mOFC|medial_orbitofrontal|orbitofrontal)"),
    ("aINS_dACC_MCC", r"(aINS|anterior_insula|insula)", r"(dACC|MCC|cingulate)"),
    ("thalamus_S1_S2_insula", r"(thalamus|S1|S2|operculum|insula)", r"(thalamus|S1|S2|operculum|insula)"),
    ("striatal_value", r"(NAc|accumbens|caudate|putamen)", r"(vmPFC|mOFC|caudate|putamen|NAc|accumbens)"),
]


def build_pathway_edges(roi_defs: pd.DataFrame, custom_edges_path: str = "") -> pd.DataFrame:
    rois = roi_defs[["roi", "roi_safe"]].drop_duplicates().copy()
    rows = []

    if custom_edges_path:
        p = Path(custom_edges_path).expanduser()
        if p.exists():
            ced = pd.read_csv(p, sep="\t")
            # Expected columns: pathway, source, target, optional directed
            for _, r in ced.iterrows():
                rows.append({
                    "pathway": r.get("pathway", "custom"),
                    "source_roi": r.get("source", r.get("source_roi", "")),
                    "target_roi": r.get("target", r.get("target_roi", "")),
                    "source_safe": sanitize_name(r.get("source", r.get("source_roi", ""))),
                    "target_safe": sanitize_name(r.get("target", r.get("target_roi", ""))),
                    "directed": bool(r.get("directed", True)),
                    "source": "custom",
                })

    # Rule-based edges based on actual ROI labels.
    for pathway, pat_a, pat_b in DEFAULT_PATHWAY_RULES:
        ra = re.compile(pat_a, flags=re.IGNORECASE)
        rb = re.compile(pat_b, flags=re.IGNORECASE)
        A = rois[rois["roi"].map(lambda x: bool(ra.search(str(x))))]
        B = rois[rois["roi"].map(lambda x: bool(rb.search(str(x))))]
        for _, a in A.iterrows():
            for _, b in B.iterrows():
                if a["roi_safe"] == b["roi_safe"]:
                    continue
                rows.append({
                    "pathway": pathway,
                    "source_roi": a["roi"],
                    "target_roi": b["roi"],
                    "source_safe": a["roi_safe"],
                    "target_safe": b["roi_safe"],
                    "directed": True,
                    "source": "rule_based",
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.drop_duplicates(["pathway", "source_safe", "target_safe"]).reset_index(drop=True)
    return out


# =============================================================================
# FC: beta-series correlation
# =============================================================================

def compute_fc_edges(trial_wide: pd.DataFrame, roi_defs: pd.DataFrame, min_trials: int, edge_mode: str, pathway_edges: pd.DataFrame) -> pd.DataFrame:
    roi_cols = [c for c in trial_wide.columns if c.startswith("roi_")]
    roi_safe = [c.replace("roi_", "", 1) for c in roi_cols]
    roi_name_map = roi_defs.set_index("roi_safe")["roi"].to_dict()

    if edge_mode == "pathway" and not pathway_edges.empty:
        undirected_pairs = set()
        for _, r in pathway_edges.iterrows():
            a, b = r["source_safe"], r["target_safe"]
            if f"roi_{a}" in roi_cols and f"roi_{b}" in roi_cols and a != b:
                undirected_pairs.add(tuple(sorted([a, b])))
        pairs = sorted(list(undirected_pairs))
    else:
        pairs = list(combinations(roi_safe, 2))

    rows = []
    for (sub, cond), d in trial_wide.groupby(["subject", "condition"]):
        if len(d) < min_trials:
            continue
        for a, b in pairs:
            ca = f"roi_{a}"
            cb = f"roi_{b}"
            if ca not in d.columns or cb not in d.columns:
                continue
            r = safe_corr(d[ca], d[cb])
            rows.append({
                "subject": sub,
                "condition": cond,
                "roi_a": roi_name_map.get(a, a),
                "roi_b": roi_name_map.get(b, b),
                "roi_a_safe": a,
                "roi_b_safe": b,
                "n_trials": int(len(d[[ca, cb]].dropna())),
                "r": r,
                "z": fisher_z(r),
            })

    fc = pd.DataFrame(rows)
    if fc.empty:
        return fc

    # Add paired contrasts per subject/edge.
    wide = fc.pivot_table(index=["subject", "roi_a", "roi_b", "roi_a_safe", "roi_b_safe"], columns="condition", values="z", aggfunc="mean").reset_index()
    wide.columns.name = None
    for c in ["passive", "up", "down"]:
        if c not in wide.columns:
            wide[c] = np.nan
    wide["up_minus_down"] = wide["up"] - wide["down"]
    wide["up_minus_passive"] = wide["up"] - wide["passive"]
    wide["down_minus_passive"] = wide["down"] - wide["passive"]
    wide["regulation_mean_minus_passive"] = 0.5 * (wide["up"] + wide["down"]) - wide["passive"]
    wide = wide.rename(columns={"passive": "z_passive", "up": "z_up", "down": "z_down"})
    long_contrast = wide.melt(
        id_vars=["subject", "roi_a", "roi_b", "roi_a_safe", "roi_b_safe"],
        value_vars=["z_passive", "z_up", "z_down", "up_minus_down", "up_minus_passive", "down_minus_passive", "regulation_mean_minus_passive"],
        var_name="quantity",
        value_name="z_value",
    )
    return long_contrast


def fc_tests(fc_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if fc_long.empty:
        return pd.DataFrame()
    for (a, b, asafe, bsafe, q), d in fc_long.groupby(["roi_a", "roi_b", "roi_a_safe", "roi_b_safe", "quantity"]):
        row = one_sample_test(d["z_value"], q)
        row.update({"roi_a": a, "roi_b": b, "roi_a_safe": asafe, "roi_b_safe": bsafe})
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = fdr(out, "p", "q_fdr_by_quantity", by=["quantity"])
    return out


# =============================================================================
# EC-lite: directed lagged beta-series predictive model
# =============================================================================

def compute_ec_lagged_edges(trial_wide: pd.DataFrame,
                            roi_defs: pd.DataFrame,
                            min_pairs: int,
                            edge_mode: str,
                            pathway_edges: pd.DataFrame) -> pd.DataFrame:
    roi_cols = [c for c in trial_wide.columns if c.startswith("roi_")]
    roi_safe = [c.replace("roi_", "", 1) for c in roi_cols]
    roi_name_map = roi_defs.set_index("roi_safe")["roi"].to_dict()

    if edge_mode == "pathway" and not pathway_edges.empty:
        directed_pairs = []
        for _, r in pathway_edges.iterrows():
            a, b = r["source_safe"], r["target_safe"]
            if f"roi_{a}" in roi_cols and f"roi_{b}" in roi_cols and a != b:
                directed_pairs.append((a, b, r.get("pathway", "pathway")))
        directed_pairs = list(dict.fromkeys(directed_pairs))
    else:
        directed_pairs = [(a, b, "all_edges") for a, b in permutations(roi_safe, 2)]

    rows = []

    # Build lagged table within each subject/run, consecutive trials.
    lag_rows = []
    for (sub, run), d in trial_wide.sort_values(["subject", "run", "trial_lss_index"]).groupby(["subject", "run"]):
        d = d.sort_values("trial_lss_index").reset_index(drop=True)
        if len(d) < 2:
            continue
        cur = d.iloc[:-1].copy().reset_index(drop=True)
        nxt = d.iloc[1:].copy().reset_index(drop=True)

        base = pd.DataFrame({
            "subject": sub,
            "run": run,
            "condition_next": nxt["condition"].values,
            "condition_current": cur["condition"].values,
            "trial_current": cur["trial_lss_index"].values,
            "trial_next": nxt["trial_lss_index"].values,
            "temperature_next": nxt.get("temperature_z_subject_lss", pd.Series(np.nan, index=nxt.index)).values,
        })
        for c in roi_cols:
            base[f"{c}_current"] = cur[c].values
            base[f"{c}_next"] = nxt[c].values
        lag_rows.append(base)

    if not lag_rows:
        return pd.DataFrame()

    lag = pd.concat(lag_rows, ignore_index=True)

    for sub, ds in lag.groupby("subject"):
        for cond, dc in ds.groupby("condition_next"):
            if len(dc) < min_pairs:
                continue
            for source, target, pathway in directed_pairs:
                sc = f"roi_{source}_current"
                tc = f"roi_{target}_current"
                tn = f"roi_{target}_next"
                if sc not in dc.columns or tc not in dc.columns or tn not in dc.columns:
                    continue
                dd = dc[[sc, tc, tn, "temperature_next", "run"]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(dd) < min_pairs:
                    continue
                if dd[sc].std(ddof=1) < 1e-12 or dd[tn].std(ddof=1) < 1e-12:
                    continue

                try:
                    # Standardize within fitted subset for comparable beta.
                    Z = dd.copy()
                    for cc in [sc, tc, tn, "temperature_next"]:
                        if Z[cc].std(ddof=1) > 1e-12:
                            Z[cc] = (Z[cc] - Z[cc].mean()) / Z[cc].std(ddof=1)
                    X = pd.DataFrame({
                        "source_current": Z[sc],
                        "target_current": Z[tc],
                        "temperature_next": Z["temperature_next"],
                    })
                    # Add run indicators when sufficient runs are available.
                    if Z["run"].nunique() > 1:
                        run_dum = pd.get_dummies(Z["run"].astype(str), prefix="run", drop_first=True, dtype=float)
                        X = pd.concat([X, run_dum], axis=1)
                    X = sm.add_constant(X, has_constant="add")
                    y = Z[tn]
                    fit = sm.OLS(y, X).fit()
                    beta = float(fit.params.get("source_current", np.nan))
                    p = float(fit.pvalues.get("source_current", np.nan))
                    rows.append({
                        "subject": sub,
                        "condition": cond,
                        "pathway": pathway,
                        "source_roi": roi_name_map.get(source, source),
                        "target_roi": roi_name_map.get(target, target),
                        "source_safe": source,
                        "target_safe": target,
                        "n_pairs": int(len(Z)),
                        "beta_source_to_target": beta,
                        "p_within_subject_model": p,
                        "model": "target_next ~ source_current + target_current + temperature_next + run",
                    })
                except Exception:
                    continue

    ec = pd.DataFrame(rows)
    if ec.empty:
        return ec

    wide = ec.pivot_table(
        index=["subject", "pathway", "source_roi", "target_roi", "source_safe", "target_safe"],
        columns="condition",
        values="beta_source_to_target",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    for c in ["passive", "up", "down"]:
        if c not in wide.columns:
            wide[c] = np.nan
    wide["up_minus_down"] = wide["up"] - wide["down"]
    wide["up_minus_passive"] = wide["up"] - wide["passive"]
    wide["down_minus_passive"] = wide["down"] - wide["passive"]
    wide["regulation_mean_minus_passive"] = 0.5 * (wide["up"] + wide["down"]) - wide["passive"]
    wide = wide.rename(columns={"passive": "ec_passive", "up": "ec_up", "down": "ec_down"})

    long = wide.melt(
        id_vars=["subject", "pathway", "source_roi", "target_roi", "source_safe", "target_safe"],
        value_vars=["ec_passive", "ec_up", "ec_down", "up_minus_down", "up_minus_passive", "down_minus_passive", "regulation_mean_minus_passive"],
        var_name="quantity",
        value_name="ec_beta",
    )
    return long


def ec_tests(ec_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if ec_long.empty:
        return pd.DataFrame()
    for (pathway, src, tgt, ss, ts, q), d in ec_long.groupby(["pathway", "source_roi", "target_roi", "source_safe", "target_safe", "quantity"]):
        row = one_sample_test(d["ec_beta"], q)
        row.update({
            "pathway": pathway,
            "source_roi": src,
            "target_roi": tgt,
            "source_safe": ss,
            "target_safe": ts,
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = fdr(out, "p", "q_fdr_by_quantity", by=["quantity"])
    return out


# =============================================================================
# Report
# =============================================================================

def write_report(outdir: Path,
                 manifest: Dict,
                 roi_tests: pd.DataFrame,
                 fc_tests_df: pd.DataFrame,
                 ec_tests_df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Trialwise ROI FC/EC report\n\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis estimates trialwise LSS activity for atlas-based ROIs, then tests condition-wise ROI activity, "
        "beta-series functional connectivity (FC), and directed lagged beta-series predictive connectivity as an EC-lite analysis.\n\n"
    )
    lines.append("## Inputs and run status\n")
    lines.append(f"- Inventory: `{manifest['inventory']}`\n")
    lines.append(f"- ROI definitions: `{manifest['roi_definitions']}`\n")
    lines.append(f"- BOLD column preference: `{manifest['bold_column']}`\n")
    lines.append(f"- Runs attempted: {manifest['runs_attempted']}\n")
    lines.append(f"- Runs OK/resumed: {manifest['runs_ok']}\n")
    lines.append(f"- Runs failed: {manifest['runs_failed']}\n")
    lines.append(f"- Trials: {manifest['n_trials_total']}\n")
    lines.append(f"- Subjects: {manifest['n_subjects']}\n")
    lines.append(f"- ROIs in output: {manifest['n_rois_output']}\n\n")

    lines.append("## EC interpretation warning\n")
    lines.append(
        "The EC table is a directed lagged predictive-connectivity analysis, not DCM. "
        "It is interpreted as an exploratory directed coupling proxy and does not establish causal or physiological directionality.\n\n"
    )

    lines.append("## Main output files\n")
    lines.append("- `tables/trialwise_roi_lss_long.tsv`\n")
    lines.append("- `tables/roi_activity_subject_tests.tsv`\n")
    lines.append("- `tables/fc_edge_contrast_tests.tsv`\n")
    lines.append("- `tables/ec_lagged_edge_contrast_tests.tsv`\n")
    lines.append("- `tables/pathway_edge_inventory.tsv`\n\n")

    # Summarize the leading findings.
    def top_lines(df, label, value_col="mean"):
        if df.empty or "q_fdr_by_quantity" not in df.columns:
            lines.append(f"## {label}\nNo results available.\n\n")
            return
        dd = df[df["quantity"].isin(["up_minus_down", "up_minus_passive", "down_minus_passive", "regulation_mean_minus_passive"])].copy()
        dd = dd[pd.to_numeric(dd["q_fdr_by_quantity"], errors="coerce") < 0.05]
        lines.append(f"## {label}: FDR-significant regulation-related findings\n")
        if dd.empty:
            lines.append("No FDR-significant regulation-related findings at q < .05.\n\n")
            return
        dd = dd.sort_values("q_fdr_by_quantity").head(20)
        for _, r in dd.iterrows():
            descriptor = ""
            if "roi" in r:
                descriptor = str(r.get("roi"))
            elif "roi_a" in r:
                descriptor = f"{r.get('roi_a')} -- {r.get('roi_b')}"
            elif "source_roi" in r:
                descriptor = f"{r.get('source_roi')} -> {r.get('target_roi')} ({r.get('pathway')})"
            lines.append(
                f"- {r.get('quantity')}: {descriptor}; mean={r.get('mean', np.nan):.4g}, "
                f"p={r.get('p', np.nan):.4g}, q={r.get('q_fdr_by_quantity', np.nan):.4g}\n"
            )
        lines.append("\n")

    top_lines(roi_tests, "ROI activity")
    top_lines(fc_tests_df, "Beta-series FC")
    top_lines(ec_tests_df, "Directed lagged EC-lite")

    (outdir / "trialwise_roi_fc_ec_report.md").write_text("".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--roi-definitions", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--behavior-trials", default="")
    ap.add_argument("--bold-column", default="bold_unsmoothed",
                    help="Preferred BOLD column. Use bold_unsmoothed for small ROIs; fallback is automatic.")
    ap.add_argument("--roi-regex", default="", help="Optional regex to include ROIs.")
    ap.add_argument("--exclude-roi-regex", default="", help="Optional regex to exclude ROIs.")
    ap.add_argument("--custom-edges", default="", help="Optional TSV with pathway/source/target directed edges.")
    ap.add_argument("--edge-mode", choices=["all", "pathway"], default="pathway",
                    help="pathway = only selected theory-driven edges for FC/EC; all = all ROI pairs.")
    ap.add_argument("--min-trials-fc", type=int, default=6)
    ap.add_argument("--min-pairs-ec", type=int, default=5)
    ap.add_argument("--hrf-model", default="spm")
    ap.add_argument("--drift-model", default="cosine")
    ap.add_argument("--high-pass", type=float, default=0.008)
    ap.add_argument("--other-mode", choices=["all_other", "by_condition"], default="all_other")
    ap.add_argument(
        "--include-rating-nuisance",
        action="store_true",
        help=(
            "Add all rating periods as one pooled HRF-convolved nuisance "
            "regressor in every LSS model."
        ),
    )
    ap.add_argument(
        "--rating-events-column",
        default="events_long",
        help="Preferred inventory column containing stimulation and rating events.",
    )
    ap.add_argument(
        "--solver-rcond",
        type=float,
        default=1e-12,
        help="Relative SVD cutoff used for nuisance residualization (default: 1e-12).",
    )
    ap.add_argument(
        "--target-unique-tol",
        type=float,
        default=1e-6,
        help="Minimum target-column residual norm relative to its original norm.",
    )
    ap.add_argument("--runs", default="", help="Optional comma-separated run numbers, e.g. 3,7.")
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    inv_path = Path(args.inventory).expanduser().resolve()
    roi_def_path = Path(args.roi_definitions).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    if not inv_path.exists():
        raise FileNotFoundError(inv_path)
    if not roi_def_path.exists():
        raise FileNotFoundError(roi_def_path)

    if outdir.exists() and args.overwrite and not args.resume:
        shutil.rmtree(outdir)
    ensure_dir(outdir)
    ensure_dir(outdir / "tables")
    ensure_dir(outdir / "per_run")
    ensure_dir(outdir / "cache_resampled_masks")

    inv = pd.read_csv(inv_path, sep="\t")
    if "status" in inv.columns:
        status = inv["status"].astype(str).str.upper()
        if (status == "READY").any():
            inv = inv[status == "READY"].copy()
        elif (status == "OK").any():
            inv = inv[status == "OK"].copy()
        else:
            inv = inv[~status.str.contains("FAIL", na=False)].copy()

    if args.runs.strip():
        requested_runs = {int(x.strip()) for x in args.runs.split(",") if x.strip()}
        if not requested_runs:
            raise ValueError("--runs was provided but no valid run numbers were parsed.")
        run_values = pd.to_numeric(inv["run"], errors="coerce")
        inv = inv[run_values.isin(requested_runs)].copy()
        if inv.empty:
            raise ValueError(f"No inventory rows matched --runs={sorted(requested_runs)}")

    if args.max_runs and args.max_runs > 0:
        inv = inv.head(args.max_runs).copy()

    roi_defs = load_roi_definitions(roi_def_path, roi_regex=args.roi_regex, exclude_regex=args.exclude_roi_regex)
    save_tsv(roi_defs, outdir / "tables" / "roi_definitions_used.tsv")

    behavior = None
    if args.behavior_trials.strip():
        bpath = Path(args.behavior_trials).expanduser().resolve()
        if not bpath.exists():
            raise FileNotFoundError(bpath)
        behavior = pd.read_csv(bpath, sep="\t")

    pathway_edges = build_pathway_edges(roi_defs, custom_edges_path=args.custom_edges)
    save_tsv(pathway_edges, outdir / "tables" / "pathway_edge_inventory.tsv")

    run_rows = []
    for _, row in inv.iterrows():
        try:
            sub, run, run_label = get_subject_run(row)
            out_run = outdir / "per_run" / f"{sub}_run-{run:02d}_roi_lss.tsv"

            if args.resume and out_run.exists() and out_run.stat().st_size > 0:
                rdf = pd.read_csv(out_run, sep="\t")
                print(f"[resume] {sub} run-{run:02d}: {len(rdf)} trials")
                run_rows.append({
                    "subject": sub, "run": run, "run_label": run_label,
                    "status": "RESUMED", "n_trials": int(len(rdf)), "out_file": str(out_run),
                })
                continue

            print(f"[fit] {sub} run-{run:02d}")
            rr = fit_lss_for_run(
                row=row,
                roi_defs=roi_defs,
                behavior=behavior,
                out_run_path=out_run,
                cache_dir=outdir / "cache_resampled_masks",
                bold_column=args.bold_column,
                hrf_model=args.hrf_model,
                drift_model=args.drift_model,
                high_pass=args.high_pass,
                other_mode=args.other_mode,
                include_rating_nuisance=args.include_rating_nuisance,
                rating_events_column=args.rating_events_column,
                solver_rcond=args.solver_rcond,
                target_unique_tol=args.target_unique_tol,
            )
            run_rows.append(rr)

        except Exception as e:
            sub = str(row.get("subject", "UNKNOWN"))
            run = row.get("run", np.nan)
            print(f"[FAILED] {sub} run={run}: {repr(e)}")
            run_rows.append({
                "subject": sub,
                "run": run,
                "run_label": row.get("run_label", ""),
                "status": "FAILED",
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })

    run_inv = pd.DataFrame(run_rows)
    save_tsv(run_inv, outdir / "tables" / "run_inventory_roi_lss.tsv")

    ok_files = []
    if "out_file" in run_inv.columns:
        for p in run_inv.loc[run_inv["status"].isin(["OK", "RESUMED"]), "out_file"].dropna():
            pp = Path(p)
            if pp.exists():
                ok_files.append(pp)

    if not ok_files:
        raise RuntimeError("No per-run ROI LSS outputs were created successfully.")

    trial_wide = pd.concat([pd.read_csv(p, sep="\t") for p in ok_files], ignore_index=True)
    trial_wide = zscore_temperature_within_subject(trial_wide)
    save_tsv(trial_wide, outdir / "tables" / "trialwise_roi_lss_wide.tsv")

    trial_long = wide_to_long(trial_wide, roi_defs)
    save_tsv(trial_long, outdir / "tables" / "trialwise_roi_lss_long.tsv")

    # ROI activity
    roi_summary = roi_activity_subject_summary(trial_long)
    save_tsv(roi_summary, outdir / "tables" / "roi_activity_subject_summary.tsv")

    roi_tests = roi_activity_tests(roi_summary)
    save_tsv(roi_tests, outdir / "tables" / "roi_activity_subject_tests.tsv")

    # FC
    fc_long = compute_fc_edges(
        trial_wide=trial_wide,
        roi_defs=roi_defs,
        min_trials=args.min_trials_fc,
        edge_mode=args.edge_mode,
        pathway_edges=pathway_edges,
    )
    save_tsv(fc_long, outdir / "tables" / "fc_beta_series_edges.tsv")

    fc_tests_df = fc_tests(fc_long)
    save_tsv(fc_tests_df, outdir / "tables" / "fc_edge_contrast_tests.tsv")

    # EC-lite
    ec_long = compute_ec_lagged_edges(
        trial_wide=trial_wide,
        roi_defs=roi_defs,
        min_pairs=args.min_pairs_ec,
        edge_mode=args.edge_mode,
        pathway_edges=pathway_edges,
    )
    save_tsv(ec_long, outdir / "tables" / "ec_lagged_edges.tsv")

    ec_tests_df = ec_tests(ec_long)
    save_tsv(ec_tests_df, outdir / "tables" / "ec_lagged_edge_contrast_tests.tsv")

    manifest = {
        "timestamp_local": now_iso(),
        "script": Path(__file__).name,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "inventory": str(inv_path),
        "inventory_sha256": sha256_file(inv_path),
        "roi_definitions": str(roi_def_path),
        "roi_definitions_sha256": sha256_file(roi_def_path),
        "behavior_trials": str(Path(args.behavior_trials).expanduser().resolve()) if args.behavior_trials.strip() else "",
        "outdir": str(outdir),
        "bold_column": args.bold_column,
        "edge_mode": args.edge_mode,
        "roi_regex": args.roi_regex,
        "exclude_roi_regex": args.exclude_roi_regex,
        "runs_attempted": int(len(run_inv)),
        "runs_ok": int(run_inv["status"].isin(["OK", "RESUMED"]).sum()),
        "runs_failed": int((run_inv["status"] == "FAILED").sum()),
        "n_trials_total": int(len(trial_wide)),
        "n_subjects": int(trial_wide["subject"].nunique()),
        "n_rois_output": int(len([c for c in trial_wide.columns if c.startswith("roi_")])),
        "n_fc_rows": int(len(fc_long)),
        "n_ec_rows": int(len(ec_long)),
        "ec_note": "EC-lite is directed lagged beta-series predictive connectivity, not DCM.",
    }
    save_json(manifest, outdir / "manifest.json")

    write_report(outdir, manifest, roi_tests, fc_tests_df, ec_tests_df)

    print("Processing completed successfully.")
    print(f"Output directory: {outdir}")
    print(f"Runs OK/resumed: {manifest['runs_ok']}")
    print(f"Runs failed: {manifest['runs_failed']}")
    print(f"Trials: {manifest['n_trials_total']}")
    print(f"ROIs: {manifest['n_rois_output']}")
    print(f"FC rows: {manifest['n_fc_rows']}")
    print(f"EC rows: {manifest['n_ec_rows']}")
    print("\nTop ROI activity tests:")
    if not roi_tests.empty:
        cols = [c for c in ["roi", "quantity", "mean", "p", "q_fdr_by_quantity", "n"] if c in roi_tests.columns]
        print(roi_tests.sort_values("q_fdr_by_quantity").head(15)[cols].to_string(index=False))
    print("\nTop FC tests:")
    if not fc_tests_df.empty:
        cols = [c for c in ["roi_a", "roi_b", "quantity", "mean", "p", "q_fdr_by_quantity", "n"] if c in fc_tests_df.columns]
        print(fc_tests_df.sort_values("q_fdr_by_quantity").head(15)[cols].to_string(index=False))
    print("\nTop EC-lite tests:")
    if not ec_tests_df.empty:
        cols = [c for c in ["pathway", "source_roi", "target_roi", "quantity", "mean", "p", "q_fdr_by_quantity", "n"] if c in ec_tests_df.columns]
        print(ec_tests_df.sort_values("q_fdr_by_quantity").head(15)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
