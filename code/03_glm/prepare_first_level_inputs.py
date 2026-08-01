#!/usr/bin/env python3
"""
prepare_first_level_inputs.py

Pre-GLM preparation for ds000140 after:
  1) fMRIPrep QC
  2) FWHM=6 mm smoothing
  3) behavioral computational modeling

Implementation notes
--------------------
Adds a balanced default spike-regressor policy:
  - motion24 + WM/CSF + top aCompCor components
  - custom one-hot spikes only for volumes with FD > threshold, default 0.5 mm
  - optional non-steady-state spikes
This avoids over-parameterizing runs with many fMRIPrep motion_outlier columns while
still modeling large framewise-displacement events.

This script creates an analysis-ready first-level input bundle. It does not run
the GLM. It prepares and validates everything needed before GLM.

Inputs
------
1. Smoothing inventory:
   derivatives/smoothed_fwhm6/tables/smoothing_inventory.tsv

2. Behavioral latent table:
   behavior_computational_analysis/tables/trialwise_behavior_latents.tsv

3. fMRIPrep confounds paths are taken from smoothing_inventory.tsv.

Outputs
-------
outdir/
  manifest.json
  pre_glm_inputs_report.md

  tables/
    first_level_run_inventory.tsv
    pre_glm_run_qc.tsv
    pre_glm_failed.tsv
    confound_strategy_summary.tsv
    sensitivity_sets.tsv

  sub-XX/run-YY/
    events_long.tsv
    events_stim.tsv
    confounds_standard.tsv
    confounds_standard.json

Conceptual policy
-----------------
- Use smoothed FWHM=6 BOLD for main univariate GLM.
- Keep unsmoothed fMRIPrep BOLD in the inventory for NPS, small ROI,
  pattern-expression, and connectivity analyses.
- Use run-level QC decisions already made; do not re-run QC here.
- Motion/nuisance are handled by:
    * subject/run QC flags from fMRIPrep QC,
    * nuisance regressors in confounds_standard.tsv,
    * pre-GLM rank/DOF checks,
    * later sensitivity analyses excluding sensitivity candidates.

Pipeline sequence
-----------------
The first-level GLM stage consumes:
  tables/first_level_run_inventory.tsv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import nibabel as nib


# =============================================================================
# General utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_local_iso() -> str:
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


def save_tsv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=False)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def bool_from_any(x) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}


def clean_subject(x) -> str:
    s = str(x)
    if s.startswith("sub-"):
        return s
    m = re.search(r"(\d+)", s)
    return f"sub-{int(m.group(1)):02d}" if m else s


def run_label(run) -> str:
    return f"run-{int(run):02d}"


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def sidecar_json_for_niigz(path: Path) -> Path:
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name.replace(".nii.gz", ".json"))
    if name.endswith(".nii"):
        return path.with_name(name.replace(".nii", ".json"))
    return path.with_suffix(".json")


def get_tr_from_json(json_path: Path) -> Optional[float]:
    if not json_path.exists():
        return None
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        tr = meta.get("RepetitionTime", None)
        return float(tr) if tr is not None else None
    except Exception:
        return None


def infer_tr_from_bold_path(bold_path: Path, default_tr: float = 2.0) -> float:
    js = sidecar_json_for_niigz(bold_path)
    tr = get_tr_from_json(js)
    return float(tr) if tr is not None else float(default_tr)


def get_nvols(path: Path) -> int:
    img = nib.load(str(path))
    if len(img.shape) != 4:
        raise ValueError(f"Expected 4D image, got {img.shape}: {path}")
    return int(img.shape[3])


# =============================================================================
# Event preparation
# =============================================================================

REQUIRED_BEHAVIOR_COLS = [
    "subject", "run", "trial_in_run", "condition",
    "temperature_z_subject", "rating_z_subject",
    "expected_pain_z", "prediction_error_z",
    "stim_onset_runrel", "rating_onset_runrel",
    "stim_duration", "rating_duration",
]


def load_behavior_latents(path: Path) -> pd.DataFrame:
    df = read_tsv(path)
    missing = [c for c in REQUIRED_BEHAVIOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Behavior latent table missing required columns: {missing}")

    df = df.copy()
    df["subject"] = df["subject"].map(clean_subject)
    df["run"] = safe_numeric(df["run"]).astype("Int64")
    df["trial_in_run"] = safe_numeric(df["trial_in_run"])
    df["condition"] = df["condition"].astype(str).str.lower().str.strip()

    numeric_cols = [
        "temperature_raw", "temperature_z_subject", "rating_raw", "rating_z_subject",
        "expected_pain_raw", "prediction_error_raw", "expected_pain_z", "prediction_error_z",
        "reg_success_raw", "reg_success_z", "abs_prediction_error_z",
        "trial_in_run_c", "run_centered_subject", "global_trial_z_subject",
        "prev_rating_z_subject", "prev_temperature_z_subject",
        "stim_onset_runrel", "rating_onset_runrel", "stim_duration", "rating_duration",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = safe_numeric(df[c])

    if "trial_id" not in df.columns:
        df["trial_id"] = df.apply(
            lambda r: f"{r['subject']}_run-{int(r['run']):02d}_trial-{int(r['trial_in_run']):02d}",
            axis=1
        )

    return df.sort_values(["subject", "run", "trial_in_run"]).reset_index(drop=True)


def build_events_for_run(drun: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return events_long and events_stim for one subject/run.

    events_long:
      - stim_passive / stim_up / stim_down events
      - rating events

    events_stim:
      - one row per thermal stimulation trial only
      - useful for parametric modulators and trial-wise behavior-to-brain models
    """
    drun = drun.sort_values("trial_in_run").copy()

    stim_rows = []
    rating_rows = []

    mod_cols = [
        "temperature_raw", "temperature_z_subject",
        "rating_raw", "rating_z_subject",
        "expected_pain_raw", "prediction_error_raw",
        "expected_pain_z", "prediction_error_z",
        "reg_success_raw", "reg_success_z",
        "abs_prediction_error_z",
        "trial_in_run_c", "run_centered_subject", "global_trial_z_subject",
        "prev_rating_z_subject", "prev_temperature_z_subject",
    ]
    mod_cols = [c for c in mod_cols if c in drun.columns]

    for _, r in drun.iterrows():
        cond = str(r["condition"])
        trial_type = f"stim_{cond}"
        base = {
            "onset": float(r["stim_onset_runrel"]),
            "duration": float(r["stim_duration"]),
            "trial_type": trial_type,
            "event_phase": "stim",
            "condition": cond,
            "trial_id": r.get("trial_id", ""),
            "trial_in_run": int(r["trial_in_run"]) if pd.notna(r["trial_in_run"]) else np.nan,
        }
        for c in mod_cols:
            base[c] = r[c]
        stim_rows.append(base)

        rating = {
            "onset": float(r["rating_onset_runrel"]),
            "duration": float(r["rating_duration"]),
            "trial_type": "rating",
            "event_phase": "rating",
            "condition": cond,
            "trial_id": r.get("trial_id", ""),
            "trial_in_run": int(r["trial_in_run"]) if pd.notna(r["trial_in_run"]) else np.nan,
        }
        # Keep modulator columns present for schema consistency, but empty for rating events.
        for c in mod_cols:
            rating[c] = np.nan
        rating_rows.append(rating)

    events_stim = pd.DataFrame(stim_rows)
    events_long = pd.concat([events_stim, pd.DataFrame(rating_rows)], ignore_index=True)
    events_long = events_long.sort_values(["onset", "event_phase", "trial_in_run"]).reset_index(drop=True)

    return events_long, events_stim


def event_timing_qc(events_long: pd.DataFrame, n_vols: int, tr: float, tolerance_s: float = 2.0) -> Dict[str, object]:
    run_duration = float(n_vols * tr)
    max_offset = float((events_long["onset"] + events_long["duration"]).max())
    min_onset = float(events_long["onset"].min())
    n_negative = int((events_long["onset"] < -1e-6).sum())
    n_after_run = int(((events_long["onset"] + events_long["duration"]) > (run_duration + tolerance_s)).sum())
    return {
        "run_duration_s": run_duration,
        "max_event_offset_s": max_offset,
        "min_event_onset_s": min_onset,
        "n_negative_onsets": n_negative,
        "n_events_after_run_plus_tolerance": n_after_run,
        "timing_ok": bool(n_negative == 0 and n_after_run == 0),
    }


# =============================================================================
# Confounds preparation
# =============================================================================

def motion24_cols() -> List[str]:
    base = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    cols = []
    cols += base
    cols += [f"{b}_derivative1" for b in base]
    cols += [f"{b}_power2" for b in base]
    cols += [f"{b}_derivative1_power2" for b in base]
    return cols


def find_acompcor_cols(conf: pd.DataFrame) -> List[str]:
    cols = []
    for c in conf.columns:
        if re.fullmatch(r"a_comp_cor_\d+", c):
            cols.append(c)
    return sorted(cols)


def custom_fd_spikes(conf: pd.DataFrame, threshold: float = 0.5) -> Tuple[pd.DataFrame, List[str]]:
    """Create one-hot spike regressors for volumes with FD > threshold."""
    if "framewise_displacement" not in conf.columns:
        return pd.DataFrame(index=conf.index), []
    fd = safe_numeric(conf["framewise_displacement"]).fillna(0.0)
    idx = np.where(fd.to_numpy(dtype=float) > float(threshold))[0]
    out = pd.DataFrame(index=conf.index)
    names = []
    for i in idx:
        name = f"fd_spike_{int(i):04d}"
        out[name] = 0.0
        out.loc[out.index[int(i)], name] = 1.0
        names.append(name)
    return out, names


def select_confounds(
    conf: pd.DataFrame,
    strategy: str,
    n_acompcor: int,
    include_cosine: bool,
    spike_mode: str,
    fd_spike_threshold: float,
    include_nonsteady: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Create a GLM-ready nuisance table.

    Recommended default:
      standard + spike_mode='fd05'
      = motion24 + white_matter + csf + top N aCompCor
        + custom FD>0.5 spike regressors
        + non-steady-state spikes.

    Why not all fMRIPrep motion_outlier* columns by default?
      In this dataset, fMRIPrep can create many outlier columns in some runs,
      leaving very low degrees of freedom. The balanced default models large
      FD events directly but avoids over-scrubbing through hundreds of columns.
      The all-spikes version can still be used as a sensitivity strategy.
    """
    strategy = strategy.lower().strip()
    spike_mode = spike_mode.lower().strip()

    chosen: List[str] = []
    missing_motion = []

    if strategy not in {"minimal", "standard", "motion24"}:
        raise ValueError("Unknown confound strategy. Use minimal, standard, or motion24.")
    if spike_mode not in {"fd05", "fd", "all", "none", "nonsteady_only"}:
        raise ValueError("Unknown spike_mode. Use fd05/fd, all, none, or nonsteady_only.")

    # Motion terms
    if strategy == "minimal":
        motion_terms = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    else:
        motion_terms = motion24_cols()

    for c in motion_terms:
        if c in conf.columns:
            chosen.append(c)
        else:
            missing_motion.append(c)

    # WM/CSF and aCompCor for the standard strategy
    if strategy == "standard":
        for c in ["white_matter", "csf"]:
            if c in conf.columns:
                chosen.append(c)

    acompcor_used = []
    if strategy == "standard" and n_acompcor > 0:
        acompcor_used = find_acompcor_cols(conf)[: int(n_acompcor)]
        chosen.extend(acompcor_used)

    # Optional fMRIPrep cosine columns
    cosine_cols = []
    if include_cosine:
        cosine_cols = sorted([c for c in conf.columns if c.startswith("cosine")])
        chosen.extend(cosine_cols)

    # Base confounds
    seen = set()
    chosen_unique = []
    for c in chosen:
        if c not in seen:
            chosen_unique.append(c)
            seen.add(c)

    if chosen_unique:
        out = conf[chosen_unique].copy()
    else:
        out = pd.DataFrame(index=conf.index)

    out = out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Spike regressors
    fmriprep_motion_spikes = sorted([c for c in conf.columns if c.startswith("motion_outlier")])
    nonsteady_cols = sorted([c for c in conf.columns if c.startswith("non_steady_state_outlier")])

    fd_spike_names: List[str] = []
    fmriprep_spikes_used: List[str] = []
    nonsteady_used: List[str] = []

    if spike_mode in {"fd05", "fd"}:
        fd_df, fd_spike_names = custom_fd_spikes(conf, threshold=fd_spike_threshold)
        out = pd.concat([out, fd_df], axis=1)

    elif spike_mode == "all":
        fmriprep_spikes_used = fmriprep_motion_spikes
        if fmriprep_spikes_used:
            out = pd.concat([out, conf[fmriprep_spikes_used].apply(pd.to_numeric, errors="coerce").fillna(0.0)], axis=1)

    elif spike_mode == "nonsteady_only":
        pass

    elif spike_mode == "none":
        include_nonsteady = False

    if include_nonsteady and nonsteady_cols:
        nonsteady_used = nonsteady_cols
        out = pd.concat([out, conf[nonsteady_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)], axis=1)

    # Remove duplicate columns if any.
    out = out.loc[:, ~out.columns.duplicated()].copy()

    # Drop all-zero columns to improve rank/DOF. This often removes nonsteady
    # columns in runs where the indicator is not actually active.
    all_zero_cols = [c for c in out.columns if np.allclose(out[c].to_numpy(dtype=float), 0)]
    out = out.drop(columns=all_zero_cols)

    meta = {
        "strategy": strategy,
        "spike_mode": spike_mode,
        "fd_spike_threshold": float(fd_spike_threshold),
        "include_nonsteady": bool(include_nonsteady),
        "n_input_rows": int(conf.shape[0]),
        "n_selected_cols_after_zero_drop": int(out.shape[1]),
        "missing_motion_terms": missing_motion,
        "acompcor_used": acompcor_used,
        "fmriprep_motion_spikes_available_n": int(len(fmriprep_motion_spikes)),
        "fmriprep_motion_spikes_used": fmriprep_spikes_used,
        "fd_spikes_used": fd_spike_names,
        "nonsteady_cols_used": nonsteady_used,
        "cosine_cols_selected": cosine_cols,
        "all_zero_cols_dropped": all_zero_cols,
        "include_cosine": bool(include_cosine),
    }

    return out, meta



def confound_matrix_qc(confounds: pd.DataFrame, n_vols: int, n_basic_task_cols: int = 4) -> Dict[str, object]:
    X = confounds.to_numpy(dtype=float) if confounds.shape[1] else np.zeros((n_vols, 0))
    if X.shape[0] != n_vols:
        return {
            "confounds_n_rows": int(X.shape[0]),
            "confounds_n_cols": int(X.shape[1]),
            "confounds_rank": np.nan,
            "confounds_condition_number": np.nan,
            "estimated_dof_after_confounds": np.nan,
            "estimated_dof_after_confounds_basic_task": np.nan,
            "confounds_rows_match_nvols": False,
        }

    if X.shape[1] == 0:
        rank = 0
        cond = np.nan
    else:
        # Standardize for condition-number diagnostic.
        Xs = X.copy()
        Xs = Xs - Xs.mean(axis=0, keepdims=True)
        sd = Xs.std(axis=0, ddof=0)
        keep = sd > 1e-12
        Xs = Xs[:, keep]
        if Xs.shape[1] == 0:
            rank = 0
            cond = np.nan
        else:
            Xs = Xs / Xs.std(axis=0, ddof=0, keepdims=True)
            rank = int(np.linalg.matrix_rank(Xs))
            try:
                cond = float(np.linalg.cond(Xs))
            except Exception:
                cond = np.nan

    dof_conf = int(n_vols - rank - 1)  # intercept
    dof_basic = int(n_vols - rank - 1 - int(n_basic_task_cols))

    return {
        "confounds_n_rows": int(X.shape[0]),
        "confounds_n_cols": int(X.shape[1]),
        "confounds_rank": int(rank),
        "confounds_condition_number": cond,
        "estimated_dof_after_confounds": dof_conf,
        "estimated_dof_after_confounds_basic_task": dof_basic,
        "confounds_rows_match_nvols": True,
    }


def pre_glm_flags(row: Dict[str, object], min_dof_warn: int, min_dof_bad: int, cond_warn: float) -> Tuple[str, str]:
    flags = []
    if not row.get("events_timing_ok", False):
        flags.append("BAD_EVENT_TIMING")
    if not row.get("confounds_rows_match_nvols", False):
        flags.append("BAD_CONFOUND_NROWS")
    dof = row.get("estimated_dof_after_confounds_basic_task", np.nan)
    if pd.notna(dof):
        if dof < min_dof_bad:
            flags.append("BAD_LOW_PREGLM_DOF")
        elif dof < min_dof_warn:
            flags.append("WARN_LOW_PREGLM_DOF")
    cond = row.get("confounds_condition_number", np.nan)
    if pd.notna(cond) and np.isfinite(cond) and cond > cond_warn:
        flags.append("WARN_HIGH_CONFOUND_COND")
    status = "OK" if not flags else ",".join(flags)
    decision = "READY" if status == "OK" or status.startswith("WARN") else "REVIEW_BEFORE_GLM"
    if "BAD" in status:
        decision = "REVIEW_BEFORE_GLM"
    return status, decision


# =============================================================================
# Load smoothing inventory
# =============================================================================

REQUIRED_SMOOTH_COLS = [
    "subject", "run", "status", "bold_unsmoothed", "bold_smoothed",
    "brain_mask", "confounds_tsv", "analysis_decision",
]


def load_smoothing_inventory(path: Path, fwhm: float) -> pd.DataFrame:
    df = read_tsv(path)
    missing = [c for c in REQUIRED_SMOOTH_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Smoothing inventory missing columns: {missing}")

    df = df.copy()
    df["subject"] = df["subject"].map(clean_subject)
    df["run"] = safe_numeric(df["run"]).astype("Int64")
    df["fwhm"] = safe_numeric(df["fwhm"])
    df["smoothing_ok"] = df["status"].astype(str).isin(["OK", "SKIPPED_EXISTS"])

    # Select requested FWHM.
    df = df[np.isclose(df["fwhm"], float(fwhm))].copy()
    df = df[df["smoothing_ok"]].copy()

    if df.empty:
        raise ValueError(f"No successful smoothing rows found for FWHM={fwhm}")

    return df.sort_values(["subject", "run"]).reset_index(drop=True)


# =============================================================================
# Main processing
# =============================================================================

def process_run(
    row: pd.Series,
    behavior: pd.DataFrame,
    outdir: Path,
    confound_strategy: str,
    n_acompcor: int,
    include_cosine: bool,
    spike_mode: str,
    fd_spike_threshold: float,
    include_nonsteady: bool,
    default_tr: float,
    min_dof_warn: int,
    min_dof_bad: int,
    cond_warn: float,
    overwrite: bool,
) -> Dict[str, object]:
    subject = str(row["subject"])
    run = int(row["run"])
    run_dir = outdir / subject / run_label(run)
    ensure_dir(run_dir)

    bold_smoothed = Path(str(row["bold_smoothed"]))
    bold_unsmoothed = Path(str(row["bold_unsmoothed"]))
    mask = Path(str(row["brain_mask"]))
    conf_path = Path(str(row["confounds_tsv"]))

    out = {
        "subject": subject,
        "run": run,
        "run_label": run_label(run),
        "bold_smoothed": str(bold_smoothed),
        "bold_unsmoothed": str(bold_unsmoothed),
        "brain_mask": str(mask),
        "confounds_raw": str(conf_path),
        "analysis_decision": row.get("analysis_decision", ""),
        "sensitivity_exclude_candidate": row.get("sensitivity_exclude_candidate", ""),
        "motion_flag": row.get("motion_flag", ""),
        "nuisance_burden_flag": row.get("nuisance_burden_flag", ""),
        "file_flag": row.get("file_flag", ""),
        "fd_mean": row.get("fd_mean", np.nan),
        "fd_max": row.get("fd_max", np.nan),
        "pct_fd_gt_0p5": row.get("pct_fd_gt_0p5", np.nan),
        "motion_outlier_volumes_pct": row.get("motion_outlier_volumes_pct", np.nan),
        "estimated_dof_after_nuisance_qc": row.get("estimated_dof_after_nuisance", np.nan),
    }

    try:
        n_vols = get_nvols(bold_smoothed)
        tr = infer_tr_from_bold_path(bold_smoothed, default_tr=default_tr)
        out["n_vols"] = n_vols
        out["tr"] = tr

        # Events
        drun = behavior[(behavior["subject"] == subject) & (behavior["run"] == run)].copy()
        if drun.empty:
            raise ValueError(f"No behavioral events found for {subject} run-{run:02d}")

        events_long, events_stim = build_events_for_run(drun)
        tqc = event_timing_qc(events_long, n_vols=n_vols, tr=tr)
        out.update({f"events_{k}": v for k, v in tqc.items()})
        out["events_timing_ok"] = tqc["timing_ok"]
        out["n_trials_behavior"] = int(drun.shape[0])
        out["n_events_long"] = int(events_long.shape[0])
        out["n_events_stim"] = int(events_stim.shape[0])

        events_long_path = run_dir / "events_long.tsv"
        events_stim_path = run_dir / "events_stim.tsv"
        if overwrite or not events_long_path.exists():
            save_tsv(events_long, events_long_path)
        if overwrite or not events_stim_path.exists():
            save_tsv(events_stim, events_stim_path)

        out["events_long"] = str(events_long_path)
        out["events_stim"] = str(events_stim_path)

        # Confounds
        conf = read_tsv(conf_path)
        conf_std, meta = select_confounds(
            conf,
            strategy=confound_strategy,
            n_acompcor=n_acompcor,
            include_cosine=include_cosine,
            spike_mode=spike_mode,
            fd_spike_threshold=fd_spike_threshold,
            include_nonsteady=include_nonsteady,
        )
        cq = confound_matrix_qc(conf_std, n_vols=n_vols, n_basic_task_cols=4)
        out.update(cq)

        conf_out = run_dir / f"confounds_{confound_strategy}.tsv"
        conf_json = run_dir / f"confounds_{confound_strategy}.json"
        if overwrite or not conf_out.exists():
            save_tsv(conf_std, conf_out)
        if overwrite or not conf_json.exists():
            save_json(meta, conf_json)

        out["confounds_glm"] = str(conf_out)
        out["confounds_glm_json"] = str(conf_json)
        out["confound_strategy"] = confound_strategy
        out["confounds_include_cosine"] = bool(include_cosine)
        out["confounds_n_acompcor"] = int(n_acompcor)
        out["confounds_spike_mode"] = spike_mode
        out["confounds_fd_spike_threshold"] = float(fd_spike_threshold)
        out["confounds_include_nonsteady"] = bool(include_nonsteady)
        out["confounds_fd_spikes_used_n"] = len(meta.get("fd_spikes_used", []))
        out["confounds_fmriprep_spikes_available_n"] = meta.get("fmriprep_motion_spikes_available_n", np.nan)
        out["confounds_fmriprep_spikes_used_n"] = len(meta.get("fmriprep_motion_spikes_used", []))
        out["confounds_nonsteady_used_n"] = len(meta.get("nonsteady_cols_used", []))

        flags, decision = pre_glm_flags(out, min_dof_warn=min_dof_warn, min_dof_bad=min_dof_bad, cond_warn=cond_warn)
        out["pre_glm_flag"] = flags
        out["pre_glm_decision"] = decision
        out["status"] = "OK"

    except Exception as e:
        out["status"] = "FAILED"
        out["error"] = repr(e)
        out["pre_glm_flag"] = "FAILED_PREP"
        out["pre_glm_decision"] = "REVIEW_BEFORE_GLM"

    return out


def write_report(
    outdir: Path,
    manifest: Dict[str, object],
    inventory: pd.DataFrame,
    failed: pd.DataFrame,
) -> None:
    n = len(inventory)
    n_ready = int((inventory["pre_glm_decision"] == "READY").sum()) if n else 0
    n_review = int((inventory["pre_glm_decision"] == "REVIEW_BEFORE_GLM").sum()) if n else 0
    n_failed = len(failed)

    lines = []
    lines.append("# Pre-GLM first-level input preparation report\n\n")
    lines.append("## Overview\n")
    lines.append(f"- Generated: {manifest['timestamp_local']}\n")
    lines.append(f"- Smoothing inventory: `{manifest['smoothing_inventory']}`\n")
    lines.append(f"- Behavioral latent table: `{manifest['behavior_latents']}`\n")
    lines.append(f"- Output directory: `{manifest['outdir']}`\n")
    lines.append(f"- FWHM selected: {manifest['fwhm']}\n")
    lines.append(f"- Confound strategy: `{manifest['confound_strategy']}`\n")
    lines.append(f"- Include fMRIPrep cosine columns: {manifest['include_cosine']}\n")
    lines.append(f"- aCompCor columns requested: {manifest['n_acompcor']}\n")
    lines.append(f"- Spike mode: `{manifest.get('spike_mode', 'NA')}`; FD spike threshold: {manifest.get('fd_spike_threshold', 'NA')}\n\n")

    lines.append("## Status\n")
    lines.append(f"- Runs prepared: {n}\n")
    lines.append(f"- READY: {n_ready}\n")
    lines.append(f"- REVIEW_BEFORE_GLM: {n_review}\n")
    lines.append(f"- FAILED: {n_failed}\n\n")

    lines.append("## Motion-control strategy\n")
    lines.append("- This step does not remove additional runs beyond the already accepted QC/smoothing inventory.\n")
    lines.append("- Motion is handled by retaining QC labels, preparing nuisance regressors, checking confound rank/DOF, and preserving sensitivity-analysis flags.\n")
    lines.append("- Main-analysis policy: include runs marked READY unless subsequent GLM design QC identifies a failure.\n")
    lines.append("- Sensitivity-analysis policy: repeat group-level tests after excluding sensitivity candidates or high-motion subjects/runs.\n\n")

    lines.append("## Event files\n")
    lines.append("- `events_long.tsv`: thermal-stimulation events plus rating-period events.\n")
    lines.append("- `events_stim.tsv`: one row per thermal-stimulation trial with behavioral latent variables.\n\n")

    lines.append("## Confounds\n")
    lines.append("- `confounds_standard.tsv` by default includes motion24, white matter, CSF, top aCompCor components, custom FD>0.5 spike regressors, and non-steady-state spikes.\n")
    lines.append("- All fMRIPrep motion_outlier columns are not included by default because they can over-parameterize low-motion runs; use `--spike-mode all` as a sensitivity strategy if needed.\n")
    lines.append("- Cosine regressors are not included by default; use GLM high-pass drift modeling to avoid double high-pass filtering.\n")
    lines.append("- The subsequent GLM stage re-evaluates design rank and exact DOF after adding task regressors and HRF convolution.\n\n")

    lines.append("## Key outputs\n")
    lines.append("- `tables/first_level_run_inventory.tsv`\n")
    lines.append("- `tables/pre_glm_run_qc.tsv`\n")
    lines.append("- `tables/pre_glm_failed.tsv`\n")
    lines.append("- `tables/confound_strategy_summary.tsv`\n")
    lines.append("- `tables/sensitivity_sets.tsv`\n")
    lines.append("- `sub-XX/run-YY/events_long.tsv`\n")
    lines.append("- `sub-XX/run-YY/events_stim.tsv`\n")
    lines.append("- `sub-XX/run-YY/confounds_standard.tsv`\n")

    (outdir / "pre_glm_inputs_report.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare first-level GLM inputs after QC, smoothing, and behavioral modeling.")
    parser.add_argument("--smoothing-inventory", required=True, type=str, help="Path to smoothed_fwhm6/tables/smoothing_inventory.tsv")
    parser.add_argument("--behavior-latents", required=True, type=str, help="Path to trialwise_behavior_latents.tsv from behavioral model")
    parser.add_argument("--outdir", required=True, type=str, help="Output directory for first-level analysis inputs")
    parser.add_argument("--fwhm", default=6.0, type=float, help="Which FWHM rows to use from smoothing inventory")
    parser.add_argument("--confound-strategy", default="standard", choices=["minimal", "standard", "motion24"])
    parser.add_argument("--n-acompcor", default=5, type=int, help="Top N aCompCor columns for standard confound strategy")
    parser.add_argument("--include-cosine", action="store_true", help="Include fMRIPrep cosine columns in confounds. Default False.")
    parser.add_argument("--spike-mode", default="fd05", choices=["fd05", "fd", "all", "none", "nonsteady_only"], help="Spike-regressor policy. Default fd05 = custom FD>threshold spikes only.")
    parser.add_argument("--fd-spike-threshold", default=0.5, type=float, help="FD threshold for custom spike regressors when spike-mode=fd05/fd.")
    parser.add_argument("--include-nonsteady", action="store_true", default=True, help="Include non-steady-state outlier columns when available. Default True.")
    parser.add_argument("--no-nonsteady", dest="include_nonsteady", action="store_false", help="Do not include non-steady-state outlier columns.")
    parser.add_argument("--default-tr", default=2.0, type=float)
    parser.add_argument("--min-dof-warn", default=80, type=int)
    parser.add_argument("--min-dof-bad", default=50, type=int)
    parser.add_argument("--cond-warn", default=1e8, type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    smoothing_path = Path(args.smoothing_inventory).expanduser().resolve()
    behavior_path = Path(args.behavior_latents).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    tables = outdir / "tables"
    ensure_dir(outdir)
    ensure_dir(tables)

    smooth = load_smoothing_inventory(smoothing_path, fwhm=args.fwhm)
    behavior = load_behavior_latents(behavior_path)

    rows = []
    for _, row in smooth.iterrows():
        rows.append(process_run(
            row=row,
            behavior=behavior,
            outdir=outdir,
            confound_strategy=args.confound_strategy,
            n_acompcor=int(args.n_acompcor),
            include_cosine=bool(args.include_cosine),
            spike_mode=str(args.spike_mode),
            fd_spike_threshold=float(args.fd_spike_threshold),
            include_nonsteady=bool(args.include_nonsteady),
            default_tr=float(args.default_tr),
            min_dof_warn=int(args.min_dof_warn),
            min_dof_bad=int(args.min_dof_bad),
            cond_warn=float(args.cond_warn),
            overwrite=bool(args.overwrite),
        ))

    inv = pd.DataFrame(rows).sort_values(["subject", "run"]).reset_index(drop=True)
    failed = inv[inv["status"] == "FAILED"].copy()
    pre_glm_qc = inv.copy()

    save_tsv(inv, tables / "first_level_run_inventory.tsv")
    save_tsv(pre_glm_qc, tables / "pre_glm_run_qc.tsv")
    save_tsv(failed, tables / "pre_glm_failed.tsv")

    # Confound strategy summary
    strategy_cols = [
        "confound_strategy", "confounds_include_cosine", "confounds_n_acompcor",
        "confounds_spike_mode", "confounds_fd_spike_threshold", "confounds_include_nonsteady",
        "confounds_fd_spikes_used_n", "confounds_fmriprep_spikes_available_n",
        "confounds_fmriprep_spikes_used_n", "confounds_nonsteady_used_n",
        "confounds_n_cols", "confounds_rank",
        "estimated_dof_after_confounds", "estimated_dof_after_confounds_basic_task",
        "pre_glm_flag", "pre_glm_decision",
    ]
    strategy_cols = [c for c in strategy_cols if c in inv.columns]
    save_tsv(inv[strategy_cols + ["subject", "run"]], tables / "confound_strategy_summary.tsv")

    # Sensitivity sets
    sens = inv[[
        "subject", "run", "analysis_decision", "sensitivity_exclude_candidate",
        "pre_glm_decision", "pre_glm_flag",
        "bold_smoothed", "bold_unsmoothed", "brain_mask", "events_long", "events_stim", "confounds_glm",
    ]].copy()
    sens["main_include"] = sens["pre_glm_decision"].eq("READY")
    sens["strict_sensitivity_include"] = sens["main_include"] & (~sens["sensitivity_exclude_candidate"].map(bool_from_any))
    save_tsv(sens, tables / "sensitivity_sets.tsv")

    manifest = {
        "timestamp_local": now_local_iso(),
        "script": Path(__file__).name,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "smoothing_inventory": str(smoothing_path),
        "smoothing_inventory_sha256": sha256_file(smoothing_path),
        "behavior_latents": str(behavior_path),
        "behavior_latents_sha256": sha256_file(behavior_path),
        "outdir": str(outdir),
        "fwhm": float(args.fwhm),
        "confound_strategy": args.confound_strategy,
        "n_acompcor": int(args.n_acompcor),
        "include_cosine": bool(args.include_cosine),
        "spike_mode": str(args.spike_mode),
        "fd_spike_threshold": float(args.fd_spike_threshold),
        "include_nonsteady": bool(args.include_nonsteady),
        "default_tr": float(args.default_tr),
        "min_dof_warn": int(args.min_dof_warn),
        "min_dof_bad": int(args.min_dof_bad),
        "cond_warn": float(args.cond_warn),
        "n_runs_input": int(smooth.shape[0]),
        "n_runs_prepared": int(inv.shape[0]),
        "n_failed": int(failed.shape[0]),
        "n_ready": int((inv["pre_glm_decision"] == "READY").sum()) if len(inv) else 0,
        "n_review_before_glm": int((inv["pre_glm_decision"] == "REVIEW_BEFORE_GLM").sum()) if len(inv) else 0,
        "notes": [
            "This script prepares first-level inputs but does not run the GLM.",
            "Confound selection and preliminary DOF/rank checks are performed here.",
            "Exact design rank and residual degrees of freedom are evaluated after task regressors and HRF convolution are added.",
            "Use smoothed BOLD for main univariate GLM and unsmoothed BOLD for pattern/ROI-sensitive analyses.",
        ],
    }
    save_json(manifest, outdir / "manifest.json")
    write_report(outdir, manifest, inv, failed)

    print("Processing completed successfully.")
    print(f"Runs input: {manifest['n_runs_input']}")
    print(f"Runs prepared: {manifest['n_runs_prepared']}")
    print(f"READY: {manifest['n_ready']}")
    print(f"REVIEW_BEFORE_GLM: {manifest['n_review_before_glm']}")
    print(f"FAILED: {manifest['n_failed']}")
    print(f"Output directory: {outdir}")
    print(f"Inventory: {tables / 'first_level_run_inventory.tsv'}")


if __name__ == "__main__":
    main()
