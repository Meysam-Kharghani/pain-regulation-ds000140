#!/usr/bin/env python3
"""
Run-level fMRIPrep confounds quality control for ds000140 or any BIDS/fMRIPrep-style dataset.

Purpose
-------
This script performs a clean, reproducible, run-level and subject-level QC pass
over fMRIPrep confounds files. It is intentionally conservative in one important
way: motion/outlier flags are separated from analysis decisions. A run can be
flagged for high nuisance burden without being automatically excluded.

Core outputs
------------
outdir/
  manifest.json
  qc_decision_rules.json
  fmriprep_qc_report.md

  tables/
    run_qc_fmriprep.tsv
    subject_qc_summary.tsv
    regulation_run_qc.tsv
    high_motion_run_candidates.tsv
    high_motion_subject_candidates.tsv
    analysis_inclusion_table.tsv

  figures/
    fig_01_flag_counts.png
    fig_02_fd_mean_by_subject.png
    fig_03_fd_mean_heatmap_subject_run.png
    fig_04_fd_pct_gt_0p5_heatmap_subject_run.png
    fig_05_regulation_runs_fd.png
    fig_06_fd_dvars_scatter.png
    fig_07_outlier_burden_by_subject.png

QC policy
---------
- Main exclusion should be reserved for severe motion, missing files, or very low
  estimated degrees of freedom.
- Moderate/high motion should usually be handled as sensitivity-exclusion
  candidates, then tested in robustness analyses.
- motion_outlier* columns are nuisance-burden indicators, not automatic deletion
  rules.

Example
-------
python run_preprocessing_qc.py \
  --fmriprep-dir /path/to/derivatives/fmriprep \
  --outdir /path/to/derivatives/qc_fmriprep_analysis \
  --reg-runs 3,7

Optional:
  --fd-warn-mean 0.20
  --fd-bad-mean 0.30
  --fd-severe-mean 0.50
  --fd-warn-pct-0p5 5
  --fd-bad-pct-0p5 10
  --fd-severe-pct-0p5 30
  --outlier-warn-pct 20
  --outlier-bad-pct 35
  --min-dof-severe 50
  --min-dof-warn 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def pct_true(mask: pd.Series, denom: Optional[int] = None) -> float:
    if denom is None:
        denom = len(mask)
    if denom == 0:
        return np.nan
    return float(mask.sum() / denom * 100.0)


def parse_int_list(x: str) -> List[int]:
    if not x:
        return []
    vals = []
    for part in str(x).split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return vals


def padded_run(run_value) -> str:
    if pd.isna(run_value):
        return ""
    try:
        return f"{int(run_value):02d}"
    except Exception:
        return str(run_value)


def entity_value_to_int(x) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        return int(str(x))
    except Exception:
        return None


# =============================================================================
# BIDS / fMRIPrep file discovery
# =============================================================================

BIDS_ENTITY_KEYS = ["sub", "ses", "task", "acq", "dir", "run", "space", "res", "desc"]


def find_confounds(fmriprep_dir: Path) -> List[Path]:
    return sorted(fmriprep_dir.rglob("*desc-confounds_timeseries.tsv"))


def parse_bids_entities(path: Path) -> Dict[str, str]:
    """Parse common BIDS entities from a filename, preserving zero padding."""
    name = path.name
    ents = {}
    for part in name.split("_"):
        if "-" not in part:
            continue
        k, v = part.split("-", 1)
        # strip suffix for  entity if needed
        v = v.replace(".tsv", "").replace(".gz", "").replace(".nii", "").replace(".json", "")
        if k in BIDS_ENTITY_KEYS:
            ents[k] = v
    return ents


def entity_match_pattern(ents: Dict[str, str], include_space: bool = True) -> str:
    parts = []
    for k in ["sub", "ses", "task", "acq", "dir", "run"]:
        if k in ents:
            parts.append(f"{k}-{ents[k]}")
    if include_space and "space" in ents:
        parts.append(f"space-{ents['space']}")
    return "_".join(parts)


def find_matching_files(confounds_tsv: Path, ents: Dict[str, str]) -> Dict[str, object]:
    """Find likely preprocessed BOLD, mask, and JSON sidecars in the same func dir."""
    func_dir = confounds_tsv.parent
    pattern_base = entity_match_pattern(ents, include_space=False)
    if not pattern_base:
        pattern_base = confounds_tsv.name.split("_desc-confounds")[0]

    bolds = sorted(func_dir.glob(f"{pattern_base}*desc-preproc_bold.nii.gz"))
    masks = sorted(func_dir.glob(f"{pattern_base}*desc-brain_mask.nii.gz"))
    bold_jsons = sorted(func_dir.glob(f"{pattern_base}*bold.json"))

    return {
        "preproc_bold_candidates_n": len(bolds),
        "brain_mask_candidates_n": len(masks),
        "bold_json_candidates_n": len(bold_jsons),
        "preproc_bold": str(bolds[0]) if bolds else "",
        "brain_mask": str(masks[0]) if masks else "",
        "bold_json": str(bold_jsons[0]) if bold_jsons else "",
    }


def get_nifti_nvols_if_possible(path_str: str) -> Optional[int]:
    """Return NIfTI 4th dimension if nibabel is available and file exists."""
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        import nibabel as nib
        img = nib.load(str(path))
        shape = img.shape
        if len(shape) >= 4:
            return int(shape[3])
        return 1
    except Exception:
        return None


def get_json_tr_if_possible(path_str: str) -> Optional[float]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        tr = meta.get("RepetitionTime", None)
        return float(tr) if tr is not None else None
    except Exception:
        return None


# =============================================================================
# Confounds summarization
# =============================================================================

def choose_dvars_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["std_dvars", "dvars"]:
        if c in df.columns:
            return c
    return None


def count_columns_by_patterns(df: pd.DataFrame) -> Dict[str, int]:
    cols = list(df.columns)
    motion6 = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    motion6_present = [c for c in motion6 if c in cols]

    motion_expanded = []
    for base in motion6:
        for c in [base, f"{base}_derivative1", f"{base}_power2", f"{base}_derivative1_power2"]:
            if c in cols:
                motion_expanded.append(c)

    cosine_cols = [c for c in cols if c.startswith("cosine")]
    acompcor_cols = [c for c in cols if re.match(r"^[at]?_?comp_cor_\d+", c)]
    motion_outlier_cols = [c for c in cols if c.startswith("motion_outlier")]
    nonsteady_cols = [c for c in cols if c.startswith("non_steady_state_outlier")]
    global_signal_cols = [c for c in ["global_signal", "global_signal_derivative1", "global_signal_power2", "global_signal_derivative1_power2"] if c in cols]
    wm_csf_cols = [
        c for c in [
            "white_matter", "white_matter_derivative1", "white_matter_power2", "white_matter_derivative1_power2",
            "csf", "csf_derivative1", "csf_power2", "csf_derivative1_power2",
        ] if c in cols
    ]

    return {
        "n_motion6_cols": len(motion6_present),
        "n_motion24_cols_present": len(motion_expanded),
        "n_cosine_cols": len(cosine_cols),
        "n_acompcor_cols": len(acompcor_cols),
        "n_motion_outlier_cols": len(motion_outlier_cols),
        "n_nonsteady_cols": len(nonsteady_cols),
        "n_global_signal_cols": len(global_signal_cols),
        "n_wm_csf_cols": len(wm_csf_cols),
    }


def summarize_confounds(tsv_path: Path, n_acompcor_for_dof: int) -> Dict[str, object]:
    df = pd.read_csv(tsv_path, sep="\t")
    n_vols = int(len(df))
    out: Dict[str, object] = {"n_vols_confounds": n_vols}

    # FD metrics
    if "framewise_displacement" in df.columns:
        fd = safe_float_series(df["framewise_displacement"])
        fd_valid = fd.dropna()
        out["fd_valid_n"] = int(fd_valid.shape[0])
        out["fd_mean"] = float(fd_valid.mean()) if len(fd_valid) else np.nan
        out["fd_median"] = float(fd_valid.median()) if len(fd_valid) else np.nan
        out["fd_p95"] = float(fd_valid.quantile(0.95)) if len(fd_valid) else np.nan
        out["fd_max"] = float(fd_valid.max()) if len(fd_valid) else np.nan
        denom = len(fd_valid) if len(fd_valid) else n_vols
        out["pct_fd_gt_0p2"] = pct_true(fd_valid > 0.2, denom=denom) if len(fd_valid) else np.nan
        out["pct_fd_gt_0p5"] = pct_true(fd_valid > 0.5, denom=denom) if len(fd_valid) else np.nan
        out["pct_fd_gt_1p0"] = pct_true(fd_valid > 1.0, denom=denom) if len(fd_valid) else np.nan
        out["n_fd_gt_0p2"] = int((fd_valid > 0.2).sum())
        out["n_fd_gt_0p5"] = int((fd_valid > 0.5).sum())
        out["n_fd_gt_1p0"] = int((fd_valid > 1.0).sum())
    else:
        for k in [
            "fd_valid_n", "fd_mean", "fd_median", "fd_p95", "fd_max",
            "pct_fd_gt_0p2", "pct_fd_gt_0p5", "pct_fd_gt_1p0",
            "n_fd_gt_0p2", "n_fd_gt_0p5", "n_fd_gt_1p0",
        ]:
            out[k] = np.nan

    # DVARS metrics
    dvars_col = choose_dvars_column(df)
    out["dvars_column_used"] = dvars_col or ""
    if dvars_col:
        dvars = safe_float_series(df[dvars_col]).dropna()
        out["dvars_valid_n"] = int(dvars.shape[0])
        out["dvars_mean"] = float(dvars.mean()) if len(dvars) else np.nan
        out["dvars_median"] = float(dvars.median()) if len(dvars) else np.nan
        out["dvars_p95"] = float(dvars.quantile(0.95)) if len(dvars) else np.nan
        out["dvars_max"] = float(dvars.max()) if len(dvars) else np.nan
    else:
        out["dvars_valid_n"] = 0
        out["dvars_mean"] = np.nan
        out["dvars_median"] = np.nan
        out["dvars_p95"] = np.nan
        out["dvars_max"] = np.nan

    counts = count_columns_by_patterns(df)
    out.update(counts)

    motion_outlier_cols = [c for c in df.columns if c.startswith("motion_outlier")]
    nonsteady_cols = [c for c in df.columns if c.startswith("non_steady_state_outlier")]

    if motion_outlier_cols:
        motion_outlier_matrix = df[motion_outlier_cols].fillna(0)
        # Usually one column per outlier volume. Sum is robust if one-hot columns.
        motion_outlier_sum = int(motion_outlier_matrix.sum().sum())
        outlier_rows = int((motion_outlier_matrix.sum(axis=1) > 0).sum())
    else:
        motion_outlier_sum = 0
        outlier_rows = 0

    if nonsteady_cols:
        nonsteady_matrix = df[nonsteady_cols].fillna(0)
        nonsteady_sum = int(nonsteady_matrix.sum().sum())
        nonsteady_rows = int((nonsteady_matrix.sum(axis=1) > 0).sum())
    else:
        nonsteady_sum = 0
        nonsteady_rows = 0

    out["motion_outliers_n_cols"] = int(len(motion_outlier_cols))
    out["motion_outliers_n_sum"] = motion_outlier_sum
    out["motion_outlier_volumes_n"] = outlier_rows
    out["motion_outlier_volumes_pct"] = float(outlier_rows / n_vols * 100.0) if n_vols else np.nan
    out["nonsteady_outliers_n_cols"] = int(len(nonsteady_cols))
    out["nonsteady_outliers_n_sum"] = nonsteady_sum
    out["nonsteady_volumes_n"] = nonsteady_rows
    out["nonsteady_volumes_pct"] = float(nonsteady_rows / n_vols * 100.0) if n_vols else np.nan

    # Estimated nuisance burden / DOF if a typical confound model is used.
    # This is not a design-rank calculation; it is a conservative run-level warning.
    n_acompcor_used = min(int(n_acompcor_for_dof), int(out["n_acompcor_cols"]))
    estimated_nuisance_cols = (
        int(out["n_motion24_cols_present"])
        + int(out["n_cosine_cols"])
        + n_acompcor_used
        + int(out["motion_outliers_n_cols"])
        + int(out["nonsteady_outliers_n_cols"])
    )
    out["n_acompcor_assumed_for_dof"] = n_acompcor_used
    out["estimated_nuisance_cols_for_dof"] = int(estimated_nuisance_cols)
    out["estimated_dof_after_nuisance"] = int(n_vols - estimated_nuisance_cols - 1)  # minus intercept

    return out


# =============================================================================
# Decision rules
# =============================================================================

def make_flags(row: pd.Series, args) -> Tuple[str, str, str]:
    motion_flags: List[str] = []
    nuisance_flags: List[str] = []
    file_flags: List[str] = []

    # File/inventory flags
    if row.get("preproc_bold_candidates_n", 0) == 0:
        file_flags.append("MISSING_PREPROC_BOLD")
    if row.get("brain_mask_candidates_n", 0) == 0:
        file_flags.append("MISSING_BRAIN_MASK")
    if pd.notna(row.get("n_vols_nifti", np.nan)) and pd.notna(row.get("n_vols_confounds", np.nan)):
        if int(row["n_vols_nifti"]) != int(row["n_vols_confounds"]):
            file_flags.append("NVOLS_MISMATCH")

    # FD flags
    fd_mean = row.get("fd_mean", np.nan)
    pct_05 = row.get("pct_fd_gt_0p5", np.nan)
    pct_10 = row.get("pct_fd_gt_1p0", np.nan)
    fd_max = row.get("fd_max", np.nan)

    if pd.notna(fd_mean):
        if fd_mean > args.fd_severe_mean:
            motion_flags.append("SEVERE_FD_MEAN")
        elif fd_mean > args.fd_bad_mean:
            motion_flags.append("BAD_FD_MEAN")
        elif fd_mean > args.fd_warn_mean:
            motion_flags.append("WARN_FD_MEAN")

    if pd.notna(pct_05):
        if pct_05 > args.fd_severe_pct_0p5:
            motion_flags.append("SEVERE_FD_GT_0.5")
        elif pct_05 > args.fd_bad_pct_0p5:
            motion_flags.append("BAD_FD_GT_0.5")
        elif pct_05 > args.fd_warn_pct_0p5:
            motion_flags.append("WARN_FD_GT_0.5")

    if pd.notna(pct_10) and pct_10 > args.fd_severe_pct_1p0:
        motion_flags.append("SEVERE_FD_GT_1.0")

    if pd.notna(fd_max):
        if fd_max > args.fd_severe_max:
            motion_flags.append("SEVERE_FD_MAX")
        elif fd_max > args.fd_warn_max:
            motion_flags.append("WARN_FD_MAX")

    # Nuisance burden flags
    out_pct = row.get("motion_outlier_volumes_pct", np.nan)
    dof = row.get("estimated_dof_after_nuisance", np.nan)

    if pd.notna(out_pct):
        if out_pct > args.outlier_bad_pct:
            nuisance_flags.append("HIGH_OUTLIER_BURDEN")
        elif out_pct > args.outlier_warn_pct:
            nuisance_flags.append("WARN_OUTLIER_BURDEN")

    if pd.notna(dof):
        if dof < args.min_dof_severe:
            nuisance_flags.append("SEVERE_LOW_ESTIMATED_DOF")
        elif dof < args.min_dof_warn:
            nuisance_flags.append("WARN_LOW_ESTIMATED_DOF")

    return (
        "OK" if not motion_flags else ",".join(motion_flags),
        "OK" if not nuisance_flags else ",".join(nuisance_flags),
        "OK" if not file_flags else ",".join(file_flags),
    )


def make_decision(row: pd.Series) -> Tuple[str, str, bool, bool]:
    """Return analysis_decision, reason, keep_main, sensitivity_exclude_candidate."""
    motion = str(row.get("motion_flag", "OK"))
    nuisance = str(row.get("nuisance_burden_flag", "OK"))
    files = str(row.get("file_flag", "OK"))

    severe_terms = [
        "MISSING_PREPROC_BOLD", "MISSING_BRAIN_MASK", "NVOLS_MISMATCH",
        "SEVERE_FD_MEAN", "SEVERE_FD_GT_0.5", "SEVERE_FD_GT_1.0",
        "SEVERE_FD_MAX",
    ]
    bad_terms = [
        "BAD_FD_MEAN", "BAD_FD_GT_0.5", "HIGH_OUTLIER_BURDEN",
        "WARN_LOW_ESTIMATED_DOF", "SEVERE_LOW_ESTIMATED_DOF",
    ]
    warn_terms = [
        "WARN_FD_MEAN", "WARN_FD_GT_0.5", "WARN_FD_MAX",
        "WARN_OUTLIER_BURDEN",
    ]

    combined = ",".join([motion, nuisance, files])

    if any(t in combined for t in severe_terms):
        return "EXCLUDE", combined, False, True

    if any(t in combined for t in bad_terms):
        return "KEEP_FOR_MAIN_BUT_SENSITIVITY_EXCLUDE_CANDIDATE", combined, True, True

    if any(t in combined for t in warn_terms):
        return "KEEP_WITH_WARNING", combined, True, False

    return "KEEP", "OK", True, False


def decision_rules_dict(args) -> Dict[str, object]:
    return {
        "fd_warn_mean": args.fd_warn_mean,
        "fd_bad_mean": args.fd_bad_mean,
        "fd_severe_mean": args.fd_severe_mean,
        "fd_warn_pct_0p5": args.fd_warn_pct_0p5,
        "fd_bad_pct_0p5": args.fd_bad_pct_0p5,
        "fd_severe_pct_0p5": args.fd_severe_pct_0p5,
        "fd_severe_pct_1p0": args.fd_severe_pct_1p0,
        "fd_warn_max": args.fd_warn_max,
        "fd_severe_max": args.fd_severe_max,
        "outlier_warn_pct": args.outlier_warn_pct,
        "outlier_bad_pct": args.outlier_bad_pct,
        "min_dof_warn": args.min_dof_warn,
        "min_dof_severe": args.min_dof_severe,
        "n_acompcor_for_dof": args.n_acompcor_for_dof,
        "main_policy": (
            "Severe file or severe FD-based motion problems are excluded. BAD_FD, high "
            "outlier burden, and low estimated DOF are retained for the main analysis but "
            "marked as sensitivity-exclusion or design-review candidates. Warnings are "
            "retained with warning."
        ),
        "important_note": (
            "motion_outlier columns indicate nuisance-regressor burden. They are not, by "
            "themselves, automatic run-exclusion rules."
        ),
    }


# =============================================================================
# Subject-level summary
# =============================================================================

def subject_summary(run_df: pd.DataFrame, reg_runs: List[int]) -> pd.DataFrame:
    """Subject-level QC summary.

    high_motion_subject_candidate is based on FD/motion flags only, not on
    nuisance/estimated-DOF burden. This keeps "high motion" distinct from
    "high nuisance burden" and avoids labeling low-motion subjects as high-motion
    merely because many censor/outlier regressors were produced by fMRIPrep.
    """
    d0 = run_df.copy()
    d0["motion_sensitivity_candidate"] = d0["motion_flag"].astype(str).str.contains("BAD|SEVERE", regex=True)
    d0["nuisance_sensitivity_candidate"] = d0["nuisance_burden_flag"].astype(str).str.contains(
        "HIGH_OUTLIER_BURDEN|LOW_ESTIMATED_DOF", regex=True
    )
    rows = []
    for sub, d in d0.groupby("subject", sort=True):
        reg = d[d["run"].isin(reg_runs)] if reg_runs else d.iloc[0:0]
        rows.append({
            "subject": sub,
            "n_runs": int(d.shape[0]),
            "n_keep_main": int(d["keep_main"].sum()),
            "n_exclude": int((d["analysis_decision"] == "EXCLUDE").sum()),
            "n_sensitivity_exclude_candidates": int(d["sensitivity_exclude_candidate"].sum()),
            "n_motion_sensitivity_candidates": int(d["motion_sensitivity_candidate"].sum()),
            "n_nuisance_sensitivity_candidates": int(d["nuisance_sensitivity_candidate"].sum()),
            "mean_fd_mean": float(d["fd_mean"].mean()),
            "median_fd_mean": float(d["fd_mean"].median()),
            "max_fd_mean": float(d["fd_mean"].max()),
            "max_fd_max": float(d["fd_max"].max()),
            "mean_pct_fd_gt_0p5": float(d["pct_fd_gt_0p5"].mean()),
            "max_pct_fd_gt_0p5": float(d["pct_fd_gt_0p5"].max()),
            "mean_motion_outlier_pct": float(d["motion_outlier_volumes_pct"].mean()),
            "min_estimated_dof": int(d["estimated_dof_after_nuisance"].min()),
            "n_reg_runs": int(reg.shape[0]),
            "n_reg_runs_keep_main": int(reg["keep_main"].sum()) if len(reg) else 0,
            "n_reg_runs_sensitivity_candidates": int(reg["sensitivity_exclude_candidate"].sum()) if len(reg) else 0,
            "n_reg_runs_motion_sensitivity_candidates": int(reg["motion_sensitivity_candidate"].sum()) if len(reg) else 0,
            "n_reg_runs_nuisance_sensitivity_candidates": int(reg["nuisance_sensitivity_candidate"].sum()) if len(reg) else 0,
            "max_reg_fd_mean": float(reg["fd_mean"].max()) if len(reg) else np.nan,
            "max_reg_pct_fd_gt_0p5": float(reg["pct_fd_gt_0p5"].max()) if len(reg) else np.nan,
        })
    out = pd.DataFrame(rows)
    out["high_motion_subject_candidate"] = (
        (out["mean_fd_mean"] > 0.25)
        | (out["n_motion_sensitivity_candidates"] >= 3)
        | (out["n_reg_runs_motion_sensitivity_candidates"] >= 1)
        | (out["n_exclude"] >= 1)
    )
    out["high_nuisance_subject_candidate"] = (
        (out["n_nuisance_sensitivity_candidates"] >= 3)
        | (out["n_reg_runs_nuisance_sensitivity_candidates"] >= 1)
        | (out["min_estimated_dof"] < 50)
    )
    return out.sort_values(["high_motion_subject_candidate", "mean_fd_mean"], ascending=[False, False]).reset_index(drop=True)


# =============================================================================
# Figures
# =============================================================================

def set_figure_defaults():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })


def fig_flag_counts(run_df: pd.DataFrame, path: Path) -> None:
    counts = run_df["analysis_decision"].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(counts)), counts.values)
    plt.xticks(np.arange(len(counts)), counts.index, rotation=30, ha="right")
    plt.ylabel("Number of runs")
    plt.title("Run-level QC decisions")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fig_fd_mean_by_subject(sub_df: pd.DataFrame, path: Path) -> None:
    d = sub_df.sort_values("mean_fd_mean")
    x = np.arange(len(d))
    plt.figure(figsize=(10, 4))
    plt.bar(x, d["mean_fd_mean"].values)
    plt.axhline(0.20, linestyle="--", linewidth=1)
    plt.axhline(0.30, linestyle="--", linewidth=1)
    plt.xticks(x, d["subject"].values, rotation=90)
    plt.ylabel("Mean of run mean FD")
    plt.title("Subject-level motion summary")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def pivot_subject_run(run_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    d = run_df.copy()
    d["run_label"] = d["run"].apply(lambda x: f"run-{int(x):02d}" if pd.notna(x) else "run-NA")
    piv = d.pivot_table(index="subject", columns="run_label", values=value_col, aggfunc="mean")
    run_cols = sorted(piv.columns, key=lambda x: int(x.split("-")[1]) if "-" in x and x.split("-")[1].isdigit() else 999)
    return piv[run_cols]


def fig_heatmap(run_df: pd.DataFrame, value_col: str, path: Path, title: str, cbar_label: str) -> None:
    piv = pivot_subject_run(run_df, value_col)
    plt.figure(figsize=(9, max(6, 0.22 * len(piv))))
    im = plt.imshow(piv.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label=cbar_label)
    plt.yticks(np.arange(len(piv.index)), piv.index)
    plt.xticks(np.arange(len(piv.columns)), piv.columns, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fig_regulation_runs_fd(run_df: pd.DataFrame, reg_runs: List[int], path: Path) -> None:
    if not reg_runs:
        return
    reg = run_df[run_df["run"].isin(reg_runs)].copy()
    if reg.empty:
        return
    reg["label"] = reg["subject"] + "_run-" + reg["run"].apply(lambda x: f"{int(x):02d}")
    reg = reg.sort_values("fd_mean")
    x = np.arange(len(reg))
    plt.figure(figsize=(12, 4.5))
    plt.bar(x, reg["fd_mean"].values)
    plt.axhline(0.20, linestyle="--", linewidth=1)
    plt.axhline(0.30, linestyle="--", linewidth=1)
    plt.xticks(x, reg["label"].values, rotation=90)
    plt.ylabel("Mean FD")
    plt.title("Mean FD in regulation runs")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fig_fd_dvars_scatter(run_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(run_df["fd_mean"], run_df["dvars_mean"])
    plt.xlabel("Mean FD")
    plt.ylabel("Mean DVARS / std DVARS")
    plt.title("Run-level FD vs DVARS")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fig_outlier_burden_by_subject(run_df: pd.DataFrame, path: Path) -> None:
    d = run_df.groupby("subject", as_index=False).agg(
        mean_outlier_pct=("motion_outlier_volumes_pct", "mean"),
        max_outlier_pct=("motion_outlier_volumes_pct", "max"),
    ).sort_values("mean_outlier_pct")
    x = np.arange(len(d))
    plt.figure(figsize=(10, 4))
    plt.bar(x, d["mean_outlier_pct"].values)
    plt.axhline(20, linestyle="--", linewidth=1)
    plt.axhline(35, linestyle="--", linewidth=1)
    plt.xticks(x, d["subject"].values, rotation=90)
    plt.ylabel("Mean motion outlier volume %")
    plt.title("Subject-level nuisance/outlier burden")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# =============================================================================
# Report
# =============================================================================

def write_report(outdir: Path, manifest: Dict[str, object], run_df: pd.DataFrame, sub_df: pd.DataFrame, reg_runs: List[int]) -> None:
    n_runs = len(run_df)
    n_subs = run_df["subject"].nunique()
    decision_counts = run_df["analysis_decision"].value_counts().to_dict()

    lines = []
    lines.append("# fMRIPrep confounds QC report\n\n")
    lines.append("## Input\n")
    lines.append(f"- Generated: {manifest['timestamp_local']}\n")
    lines.append(f"- fMRIPrep directory: `{manifest['fmriprep_dir']}`\n")
    lines.append(f"- Number of confounds files: {n_runs}\n")
    lines.append(f"- Number of subjects: {n_subs}\n")
    lines.append(f"- Regulation runs: {','.join(map(str, reg_runs)) if reg_runs else 'not specified'}\n\n")

    lines.append("## Decision counts\n")
    for k, v in sorted(decision_counts.items()):
        lines.append(f"- {k}: {v}\n")
    lines.append("\n")

    lines.append("## Motion summary\n")
    lines.append(f"- Mean FD across runs: mean={run_df['fd_mean'].mean():.4f}, median={run_df['fd_mean'].median():.4f}, max={run_df['fd_mean'].max():.4f}\n")
    lines.append(f"- Percent FD>0.5 across runs: mean={run_df['pct_fd_gt_0p5'].mean():.2f}%, median={run_df['pct_fd_gt_0p5'].median():.2f}%, max={run_df['pct_fd_gt_0p5'].max():.2f}%\n")
    lines.append(f"- Runs with mean FD > 0.30: {int((run_df['fd_mean'] > 0.30).sum())} / {n_runs}\n")
    lines.append(f"- Runs with >10% volumes FD>0.5: {int((run_df['pct_fd_gt_0p5'] > 10).sum())} / {n_runs}\n\n")

    if reg_runs:
        reg = run_df[run_df["run"].isin(reg_runs)].copy()
        lines.append("## Regulation-run motion summary\n")
        lines.append(f"- Regulation runs found: {len(reg)}\n")
        if len(reg):
            lines.append(f"- Mean FD in regulation runs: mean={reg['fd_mean'].mean():.4f}, median={reg['fd_mean'].median():.4f}, max={reg['fd_mean'].max():.4f}\n")
            lines.append(f"- Regulation runs marked as sensitivity candidates: {int(reg['sensitivity_exclude_candidate'].sum())} / {len(reg)}\n")
            lines.append(f"- Regulation runs excluded from main: {int((reg['analysis_decision'] == 'EXCLUDE').sum())} / {len(reg)}\n\n")

    lines.append("## High-motion subject candidates\n")
    hs = sub_df[sub_df["high_motion_subject_candidate"]].copy()
    if hs.empty:
        lines.append("- None by the current rules.\n\n")
    else:
        for _, r in hs.iterrows():
            lines.append(
                f"- {r['subject']}: mean FD={r['mean_fd_mean']:.3f}, "
                f"motion-candidate runs={int(r['n_motion_sensitivity_candidates'])}, "
                f"reg motion-candidate runs={int(r['n_reg_runs_motion_sensitivity_candidates'])}\n"
            )
        lines.append("\n")

    hn = sub_df[sub_df.get("high_nuisance_subject_candidate", False)].copy()
    lines.append("## High nuisance-burden subject candidates\n")
    if hn.empty:
        lines.append("- None by the current rules.\n\n")
    else:
        for _, r in hn.iterrows():
            lines.append(
                f"- {r['subject']}: nuisance-candidate runs={int(r['n_nuisance_sensitivity_candidates'])}, "
                f"min estimated DOF={int(r['min_estimated_dof'])}\n"
            )
        lines.append("\n")

    lines.append("## Interpretation policy\n")
    lines.append("- `EXCLUDE marks severe file or FD-based motion problems; low estimated DOF is retained for main analyses but flagged for design review and sensitivity analyses.\n")
    lines.append("- `KEEP_FOR_MAIN_BUT_SENSITIVITY_EXCLUDE_CANDIDATE` means the run remains eligible for the main analysis and is excluded in the corresponding sensitivity analysis.\n")
    lines.append("- `KEEP_WITH_WARNING` marks mild-to-moderate motion or nuisance burden.\n")
    lines.append("- `motion_outlier*` columns are interpreted as nuisance-regressor burden, not as automatic deletion criteria.\n\n")

    lines.append("## Key outputs\n")
    lines.append("- `tables/run_qc_fmriprep.tsv`\n")
    lines.append("- `tables/subject_qc_summary.tsv`\n")
    lines.append("- `tables/regulation_run_qc.tsv`\n")
    lines.append("- `tables/high_motion_run_candidates.tsv`\n")
    lines.append("- `tables/high_motion_subject_candidates.tsv`\n")
    lines.append("- `tables/analysis_inclusion_table.tsv`\n")
    lines.append("- `figures/`\n")

    (outdir / "fmriprep_qc_report.md").write_text("".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="analysis-level fMRIPrep confounds QC.")
    parser.add_argument("--fmriprep-dir", required=True, type=str, help="Path to fMRIPrep derivatives directory")
    parser.add_argument("--outdir", required=True, type=str, help="Output directory for QC results")
    parser.add_argument("--reg-runs", default="3,7", type=str, help="Comma-separated regulation run numbers, e.g. 3,7")
    parser.add_argument("--n-acompcor-for-dof", default=5, type=int, help="Number of aCompCor columns assumed in estimated DOF burden")

    # FD thresholds
    parser.add_argument("--fd-warn-mean", default=0.20, type=float)
    parser.add_argument("--fd-bad-mean", default=0.30, type=float)
    parser.add_argument("--fd-severe-mean", default=0.50, type=float)
    parser.add_argument("--fd-warn-pct-0p5", default=5.0, type=float)
    parser.add_argument("--fd-bad-pct-0p5", default=10.0, type=float)
    parser.add_argument("--fd-severe-pct-0p5", default=30.0, type=float)
    parser.add_argument("--fd-severe-pct-1p0", default=10.0, type=float)
    parser.add_argument("--fd-warn-max", default=2.0, type=float)
    parser.add_argument("--fd-severe-max", default=5.0, type=float)

    # nuisance/DOF thresholds
    parser.add_argument("--outlier-warn-pct", default=20.0, type=float)
    parser.add_argument("--outlier-bad-pct", default=35.0, type=float)
    parser.add_argument("--min-dof-warn", default=100, type=int)
    parser.add_argument("--min-dof-severe", default=50, type=int)

    args = parser.parse_args()

    fmriprep_dir = Path(args.fmriprep_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    tables = outdir / "tables"
    figures = outdir / "figures"
    ensure_dir(outdir)
    ensure_dir(tables)
    ensure_dir(figures)
    set_figure_defaults()

    reg_runs = parse_int_list(args.reg_runs)

    files = find_confounds(fmriprep_dir)
    if not files:
        raise SystemExit(f"No fMRIPrep confounds TSV files found under: {fmriprep_dir}")

    rows = []
    for f in files:
        ents = parse_bids_entities(f)
        matching = find_matching_files(f, ents)
        summ = summarize_confounds(f, n_acompcor_for_dof=args.n_acompcor_for_dof)

        sub_label = f"sub-{ents.get('sub', '')}" if "sub" in ents else ""
        run_int = entity_value_to_int(ents.get("run", None))

        row = {
            "confounds_tsv": str(f),
            "confounds_sha256": sha256_file(f),
            "subject": sub_label,
            "session": f"ses-{ents['ses']}" if "ses" in ents else "",
            "task": ents.get("task", ""),
            "run": run_int,
            "run_label": f"run-{padded_run(run_int)}" if run_int is not None else "",
            "acq": ents.get("acq", ""),
            "dir": ents.get("dir", ""),
            **matching,
            **summ,
        }

        nvols_nifti = get_nifti_nvols_if_possible(row["preproc_bold"])
        tr_json = get_json_tr_if_possible(row["bold_json"])
        row["n_vols_nifti"] = nvols_nifti if nvols_nifti is not None else np.nan
        row["tr_from_bold_json"] = tr_json if tr_json is not None else np.nan

        rows.append(row)

    run_df = pd.DataFrame(rows)

    # Sort cleanly
    run_df = run_df.sort_values(["subject", "session", "task", "run"], na_position="last").reset_index(drop=True)

    # Flags and decisions
    flags = run_df.apply(lambda r: make_flags(r, args), axis=1)
    run_df["motion_flag"] = [x[0] for x in flags]
    run_df["nuisance_burden_flag"] = [x[1] for x in flags]
    run_df["file_flag"] = [x[2] for x in flags]

    decisions = run_df.apply(make_decision, axis=1)
    run_df["analysis_decision"] = [x[0] for x in decisions]
    run_df["decision_reason"] = [x[1] for x in decisions]
    run_df["keep_main"] = [x[2] for x in decisions]
    run_df["sensitivity_exclude_candidate"] = [x[3] for x in decisions]

    # Tables
    save_tsv(run_df, tables / "run_qc_fmriprep.tsv")

    reg_df = run_df[run_df["run"].isin(reg_runs)].copy() if reg_runs else run_df.iloc[0:0].copy()
    save_tsv(reg_df, tables / "regulation_run_qc.tsv")

    high_run = run_df[
        (run_df["analysis_decision"] == "EXCLUDE")
        | (run_df["sensitivity_exclude_candidate"])
    ].copy()
    save_tsv(high_run, tables / "high_motion_run_candidates.tsv")

    sub_df = subject_summary(run_df, reg_runs=reg_runs)
    save_tsv(sub_df, tables / "subject_qc_summary.tsv")
    save_tsv(sub_df[sub_df["high_motion_subject_candidate"]].copy(), tables / "high_motion_subject_candidates.tsv")

    inclusion_cols = [
        "subject", "session", "task", "run", "run_label",
        "keep_main", "sensitivity_exclude_candidate", "analysis_decision",
        "motion_flag", "nuisance_burden_flag", "file_flag",
        "fd_mean", "fd_max", "pct_fd_gt_0p5", "motion_outlier_volumes_pct",
        "estimated_dof_after_nuisance",
        "preproc_bold", "brain_mask", "confounds_tsv",
    ]
    inclusion_cols = [c for c in inclusion_cols if c in run_df.columns]
    save_tsv(run_df[inclusion_cols], tables / "analysis_inclusion_table.tsv")

    rules = decision_rules_dict(args)
    save_json(rules, outdir / "qc_decision_rules.json")

    # Figures
    fig_flag_counts(run_df, figures / "fig_01_flag_counts.png")
    fig_fd_mean_by_subject(sub_df, figures / "fig_02_fd_mean_by_subject.png")
    fig_heatmap(run_df, "fd_mean", figures / "fig_03_fd_mean_heatmap_subject_run.png", "Mean FD by subject and run", "Mean FD")
    fig_heatmap(run_df, "pct_fd_gt_0p5", figures / "fig_04_fd_pct_gt_0p5_heatmap_subject_run.png", "% volumes FD > 0.5 by subject and run", "% FD > 0.5")
    fig_regulation_runs_fd(run_df, reg_runs, figures / "fig_05_regulation_runs_fd.png")
    fig_fd_dvars_scatter(run_df, figures / "fig_06_fd_dvars_scatter.png")
    fig_outlier_burden_by_subject(run_df, figures / "fig_07_outlier_burden_by_subject.png")

    manifest = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "fmriprep_dir": str(fmriprep_dir),
        "outdir": str(outdir),
        "n_confounds_files": int(len(files)),
        "n_subjects": int(run_df["subject"].nunique()),
        "reg_runs": reg_runs,
        "decision_rules": rules,
    }
    save_json(manifest, outdir / "manifest.json")

    write_report(outdir, manifest, run_df, sub_df, reg_runs)

    print("Processing completed successfully.")
    print(f"Confounds files: {len(files)}")
    print(f"Subjects: {run_df['subject'].nunique()}")
    print(f"Output directory: {outdir}")
    print("\nDecision counts:")
    print(run_df["analysis_decision"].value_counts().to_string())


if __name__ == "__main__":
    main()
