#!/usr/bin/env python3
"""
run_first_level_glm.py

Analysis-level first-level GLM for ds000140 using standardized pre-GLM inputs.

Implementation notes
--------------------
- Robust report writing when runs are skipped/resumed or failed.
- If completed.ok exists, loads existing run_qc.json and discovers existing contrast maps.
- Initializes design-QC fields before fitting to avoid missing-column crashes.
- Counts `SKIPPED_DONE` runs as completed in the generated report.

This script assumes the following preprocessing steps are complete:
  1) fMRIPrep QC
  2) FWHM=6 smoothing
  3) behavioral computational modeling
  4) pre-GLM input preparation

Main input
----------
first_level_inputs_analysis/tables/first_level_run_inventory.tsv

Main model
----------
For each run:
  - stimulation condition regressor:
      stim_passive OR stim_up OR stim_down
  - optional within-run centered temperature parametric modulator:
      stim_passive_temp OR stim_up_temp OR stim_down_temp
  - rating-period nuisance regressor:
      rating
  - nuisance confounds from pre-GLM:
      motion24 + WM/CSF + top 5 aCompCor + FD>0.5 custom spikes + nonsteady spikes

GLM settings
------------
Default settings are chosen to match this project:
  - BOLD: FWHM=6 mm smoothed image from smoothed_fwhm6
  - Mask: space-matched fMRIPrep mask
  - HRF: SPM + derivative
  - Drift: cosine
  - High-pass: 180 s cutoff
  - Noise: AR(1)
  - Signal scaling: 0 (Nilearn scaling along time, PSC-like)

Outputs
-------
outdir/
  manifest.json
  first_level_glm_report.md

  tables/
    first_level_run_inventory.tsv
    run_contrast_inventory.tsv
    subject_fixed_effects_inventory.tsv
    group_input_*.tsv
    first_level_failed.tsv

  sub-XX/run-YY/
    design_matrix.tsv
    design_matrix.png
    design_corr.png
    events_glm.tsv
    confounds_used.tsv
    run_qc.json
    beta_<contrast>.nii.gz
    var_<contrast>.nii.gz
    stat_<contrast>.nii.gz
    z_<contrast>.nii.gz

  sub-XX/subject_fixed_effects/
    beta_stim_passive_FE.nii.gz
    beta_stim_up_FE.nii.gz
    beta_stim_down_FE.nii.gz
    beta_up_minus_down_FE.nii.gz
    beta_up_minus_passive_FE.nii.gz
    beta_down_minus_passive_FE.nii.gz
    beta_regulation_mean_minus_passive_FE.nii.gz
    corresponding var/stat/z maps

Important interpretation
------------------------
- This first-level model estimates run-level condition effects. Because each
  regulation run contains only one regulation direction, up/down/passive
  contrasts are constructed at the subject fixed-effects stage.
- Temperature pmods are centered within run; the condition regressor therefore
  reflects the mean stimulation response within that run/condition.
- Behavioral latent variables are not included as default GLM regressors here.
  They are retained in events_stim.tsv for later brain-behavior or trial-wise
  modeling. This avoids overfitting the small regulation runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix


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


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def zscore_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - m) / sd


def is_binary_or_constant(x: np.ndarray) -> bool:
    vals = np.unique(np.asarray(x)[np.isfinite(x)])
    if vals.size <= 1:
        return True
    if vals.size <= 2 and set(vals.tolist()).issubset({0.0, 1.0}):
        return True
    return False


def clean_subject(x) -> str:
    s = str(x)
    if s.startswith("sub-"):
        return s
    m = re.search(r"(\d+)", s)
    return f"sub-{int(m.group(1)):02d}" if m else s


def run_label(run: int) -> str:
    return f"run-{int(run):02d}"


def matrix_rank_and_condition(X: np.ndarray) -> Tuple[int, float]:
    """Rank/condition diagnostic after removing constant columns and standardizing."""
    if X.size == 0:
        return 0, np.nan
    X = np.asarray(X, dtype=float)
    X = X[:, np.all(np.isfinite(X), axis=0)]
    if X.shape[1] == 0:
        return 0, np.nan

    Xc = X - X.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, ddof=0)
    keep = sd > 1e-12
    Xc = Xc[:, keep]
    if Xc.shape[1] == 0:
        return 0, np.nan

    Xz = Xc / Xc.std(axis=0, ddof=0, keepdims=True)
    try:
        rank = int(np.linalg.matrix_rank(Xz))
    except Exception:
        rank = np.nan
    try:
        cond = float(np.linalg.cond(Xz))
    except Exception:
        cond = np.nan
    return rank, cond


def sidecar_json_for_niigz(path: Path) -> Path:
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name.replace(".nii.gz", ".json"))
    if name.endswith(".nii"):
        return path.with_name(name.replace(".nii", ".json"))
    return path.with_suffix(".json")


def get_tr_from_json_or_default(bold_path: Path, default_tr: float = 2.0) -> float:
    js = sidecar_json_for_niigz(bold_path)
    if js.exists():
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
            if "RepetitionTime" in meta:
                return float(meta["RepetitionTime"])
        except Exception:
            pass
    try:
        img = nib.load(str(bold_path))
        zooms = img.header.get_zooms()
        if len(zooms) >= 4 and np.isfinite(zooms[3]) and zooms[3] > 0:
            return float(zooms[3])
    except Exception:
        pass
    return float(default_tr)


def get_nvols(path: Path) -> int:
    img = nib.load(str(path))
    if len(img.shape) != 4:
        raise ValueError(f"Expected 4D image: {path}; shape={img.shape}")
    return int(img.shape[3])


# =============================================================================
# Inputs
# =============================================================================

REQUIRED_INVENTORY_COLS = [
    "subject", "run", "bold_smoothed", "bold_unsmoothed", "brain_mask",
    "events_long", "events_stim", "confounds_glm",
    "pre_glm_decision", "status",
]


def load_firstlevel_inventory(path: Path, include_set: str = "ready") -> pd.DataFrame:
    inv = read_tsv(path)
    missing = [c for c in REQUIRED_INVENTORY_COLS if c not in inv.columns]
    if missing:
        raise ValueError(f"First-level input inventory missing required columns: {missing}")

    inv = inv.copy()
    inv["subject"] = inv["subject"].map(clean_subject)
    inv["run"] = safe_numeric(inv["run"]).astype("Int64")
    inv["status"] = inv["status"].astype(str)
    inv["pre_glm_decision"] = inv["pre_glm_decision"].astype(str)

    include_set = include_set.lower().strip()
    if include_set == "ready":
        inv = inv[(inv["status"] == "OK") & (inv["pre_glm_decision"] == "READY")].copy()
    elif include_set == "all":
        inv = inv.copy()
    else:
        raise ValueError("include_set must be 'ready' or 'all'.")

    return inv.sort_values(["subject", "run"]).reset_index(drop=True)


# =============================================================================
# GLM event construction
# =============================================================================

def build_glm_events(
    events_stim_path: Path,
    events_long_path: Path,
    include_rating: bool = True,
    include_temperature_pmod: bool = True,
    pmod_column: str = "temperature_z_subject",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    stim = read_tsv(events_stim_path)
    long = read_tsv(events_long_path)

    rows: List[Dict[str, object]] = []
    qc: Dict[str, object] = {
        "events_stim_path": str(events_stim_path),
        "events_long_path": str(events_long_path),
        "n_stim_trials": int(stim.shape[0]),
        "include_rating": bool(include_rating),
        "include_temperature_pmod": bool(include_temperature_pmod),
        "pmod_column": pmod_column,
        "pmod_included": False,
    }

    # Stimulation condition mean regressors.
    for _, r in stim.iterrows():
        cond = str(r["condition"]).lower().strip()
        rows.append({
            "onset": float(r["onset"]),
            "duration": float(r["duration"]),
            "trial_type": f"stim_{cond}",
            "modulation": 1.0,
        })

    # Temperature parametric modulator, centered within run.
    if include_temperature_pmod and pmod_column in stim.columns:
        vals = safe_numeric(stim[pmod_column])
        finite = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite) >= 2 and float(finite.std(ddof=0)) > 1e-10:
            centered = vals - float(finite.mean())
            qc["pmod_mean_before_centering"] = float(finite.mean())
            qc["pmod_sd_before_centering"] = float(finite.std(ddof=0))
            qc["pmod_sd_after_centering"] = float(centered.std(ddof=0))
            for idx, r in stim.iterrows():
                mod = centered.loc[idx]
                if pd.isna(mod):
                    continue
                cond = str(r["condition"]).lower().strip()
                rows.append({
                    "onset": float(r["onset"]),
                    "duration": float(r["duration"]),
                    "trial_type": f"stim_{cond}_temp",
                    "modulation": float(mod),
                })
            qc["pmod_included"] = True
        else:
            qc["pmod_skip_reason"] = "missing_or_constant_pmod"
    elif include_temperature_pmod:
        qc["pmod_skip_reason"] = f"pmod_column_missing:{pmod_column}"

    # Rating period nuisance.
    if include_rating:
        if "event_phase" in long.columns:
            rating = long[long["event_phase"].astype(str) == "rating"].copy()
        else:
            rating = long[long["trial_type"].astype(str) == "rating"].copy()
        for _, r in rating.iterrows():
            rows.append({
                "onset": float(r["onset"]),
                "duration": float(r["duration"]),
                "trial_type": "rating",
                "modulation": 1.0,
            })
        qc["n_rating_events"] = int(rating.shape[0])
    else:
        qc["n_rating_events"] = 0

    events = pd.DataFrame(rows)
    events = events.sort_values(["onset", "trial_type"]).reset_index(drop=True)
    qc["n_events_glm"] = int(events.shape[0])
    qc["trial_types"] = sorted(events["trial_type"].unique().tolist())

    return events, qc


# =============================================================================
# Confounds
# =============================================================================

def load_and_clean_confounds(path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    conf = read_tsv(path)
    conf = conf.copy()

    # Numeric, finite, fill.
    for c in conf.columns:
        conf[c] = pd.to_numeric(conf[c], errors="coerce")
    conf = conf.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    dropped_all_zero = []
    dropped_constant = []
    cleaned_cols = []

    for c in conf.columns:
        x = conf[c].to_numpy(dtype=float)
        if np.allclose(x, 0):
            dropped_all_zero.append(c)
            continue
        if is_binary_or_constant(x):
            # Retain binary spike regressors and remove nonzero constant columns.
            vals = np.unique(x[np.isfinite(x)])
            if vals.size <= 1:
                dropped_constant.append(c)
                continue
            cleaned_cols.append(c)
        else:
            conf[c] = zscore_vec(x)
            cleaned_cols.append(c)

    conf = conf[cleaned_cols].copy()

    rank, cond = matrix_rank_and_condition(conf.to_numpy(dtype=float)) if conf.shape[1] else (0, np.nan)
    qc = {
        "confounds_input": str(path),
        "n_confounds_input": int(len(cleaned_cols) + len(dropped_all_zero) + len(dropped_constant)),
        "n_confounds_used": int(conf.shape[1]),
        "dropped_all_zero": dropped_all_zero,
        "dropped_constant": dropped_constant,
        "confounds_rank": int(rank),
        "confounds_condition_number": cond,
    }
    return conf, qc


# =============================================================================
# Plotting
# =============================================================================

def plot_design_matrix_safe(design: pd.DataFrame, out_png: Path) -> None:
    try:
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111)
        plot_design_matrix(design, ax=ax)
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
    except Exception as e:
        out_png.with_suffix(".error.txt").write_text(repr(e), encoding="utf-8")


def plot_design_corr_safe(design: pd.DataFrame, out_png: Path) -> None:
    try:
        X = design.to_numpy(dtype=float)
        if X.shape[1] <= 1:
            return
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(X, rowvar=False)
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111)
        im = ax.imshow(C, aspect="auto", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_title("Design matrix correlation")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
    except Exception as e:
        out_png.with_suffix(".error.txt").write_text(repr(e), encoding="utf-8")


# =============================================================================
# Contrast helpers
# =============================================================================

def target_contrasts_from_design(design_cols: Sequence[str]) -> List[str]:
    """Run-level single-regressor effects to save if present."""
    base_targets = [
        "stim_passive", "stim_up", "stim_down",
        "stim_passive_temp", "stim_up_temp", "stim_down_temp",
        "rating",
    ]
    return [c for c in base_targets if c in design_cols]


def compute_and_save_contrast(model: FirstLevelModel, contrast_name: str, out_dir: Path) -> Dict[str, object]:
    """Save effect, variance, stat, and z if possible for one contrast expression."""
    result = {"contrast": contrast_name, "ok": False, "error": "", "maps": {}}
    safe_name = contrast_name.replace(" ", "").replace("+", "plus").replace("-", "minus").replace("*", "x").replace("/", "_")

    try:
        beta = model.compute_contrast(contrast_name, output_type="effect_size")
        beta_path = out_dir / f"beta_{safe_name}.nii.gz"
        beta.to_filename(str(beta_path))
        result["maps"]["beta"] = str(beta_path)

        z = model.compute_contrast(contrast_name, output_type="z_score")
        z_path = out_dir / f"z_{safe_name}.nii.gz"
        z.to_filename(str(z_path))
        result["maps"]["z"] = str(z_path)

        stat = model.compute_contrast(contrast_name, output_type="stat")
        stat_path = out_dir / f"stat_{safe_name}.nii.gz"
        stat.to_filename(str(stat_path))
        result["maps"]["stat"] = str(stat_path)

        try:
            var = model.compute_contrast(contrast_name, output_type="effect_variance")
            var_path = out_dir / f"var_{safe_name}.nii.gz"
            var.to_filename(str(var_path))
            result["maps"]["var"] = str(var_path)
        except Exception as e:
            result["maps"]["var"] = ""
            result["var_error"] = repr(e)

        result["ok"] = True
    except Exception as e:
        result["ok"] = False
        result["error"] = repr(e)

    return result



def strip_nii_gz_name(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def discover_existing_contrast_rows(subject: str, run: int, run_dir: Path) -> List[Dict[str, object]]:
    """Discover already-computed contrast maps in a run directory.

    Used when --resume finds completed.ok. This permits reconstruction of the global
    contrast inventory without refitting all runs.
    """
    rows = []
    for beta in sorted(run_dir.glob("beta_*.nii.gz")):
        stem = strip_nii_gz_name(beta)
        contrast = stem.replace("beta_", "", 1)
        var = run_dir / f"var_{contrast}.nii.gz"
        stat = run_dir / f"stat_{contrast}.nii.gz"
        z = run_dir / f"z_{contrast}.nii.gz"
        rows.append({
            "subject": subject,
            "run": int(run),
            "contrast": contrast,
            "ok": True,
            "error": "",
            "beta_map": str(beta),
            "var_map": str(var) if var.exists() else "",
            "stat_map": str(stat) if stat.exists() else "",
            "z_map": str(z) if z.exists() else "",
        })
    return rows


# =============================================================================
# Run-level GLM
# =============================================================================

def fit_one_run(
    row: pd.Series,
    outdir: Path,
    args,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    subject = str(row["subject"])
    run = int(row["run"])
    rlabel = run_label(run)
    run_dir = outdir / subject / rlabel
    ensure_dir(run_dir)

    bold = Path(str(row["bold_smoothed"]))
    bold_unsmoothed = Path(str(row["bold_unsmoothed"]))
    mask = Path(str(row["brain_mask"]))
    events_stim = Path(str(row["events_stim"]))
    events_long = Path(str(row["events_long"]))
    confounds_path = Path(str(row["confounds_glm"]))

    run_qc: Dict[str, object] = {
        "subject": subject,
        "run": run,
        "run_label": rlabel,
        "status": "UNKNOWN",
        "bold": str(bold),
        "bold_unsmoothed": str(bold_unsmoothed),
        "mask": str(mask),
        "events_stim": str(events_stim),
        "events_long": str(events_long),
        "confounds_glm": str(confounds_path),
        "analysis_decision": row.get("analysis_decision", ""),
        "sensitivity_exclude_candidate": row.get("sensitivity_exclude_candidate", ""),
        "motion_flag": row.get("motion_flag", ""),
        "nuisance_burden_flag": row.get("nuisance_burden_flag", ""),
        "fd_mean": row.get("fd_mean", np.nan),
        "fd_max": row.get("fd_max", np.nan),
        "pct_fd_gt_0p5": row.get("pct_fd_gt_0p5", np.nan),
        "motion_outlier_volumes_pct": row.get("motion_outlier_volumes_pct", np.nan),
        "error": "",
        "n_vols": np.nan,
        "tr": np.nan,
        "design_n_rows": np.nan,
        "design_n_cols": np.nan,
        "design_rank": np.nan,
        "design_condition_number": np.nan,
        "df_resid_approx": np.nan,
        "warnings_n": np.nan,
        "design_flags": "NOT_FIT",
        "n_contrasts_requested": 0,
        "n_contrasts_ok": 0,
        "skipped_existing": False,
    }

    contrast_rows: List[Dict[str, object]] = []

    sentinel = run_dir / "completed.ok"
    if args.resume and sentinel.exists() and not args.overwrite:
        qc_path = run_dir / "run_qc.json"
        if qc_path.exists():
            try:
                old_qc = json.loads(qc_path.read_text(encoding="utf-8"))
                if isinstance(old_qc, dict):
                    run_qc.update(old_qc)
            except Exception:
                pass
        run_qc["status"] = "SKIPPED_DONE"
        run_qc["skipped_existing"] = True
        contrast_rows = discover_existing_contrast_rows(subject, run, run_dir)
        run_qc["n_contrasts_ok"] = int(sum(1 for r in contrast_rows if r.get("ok")))
        return run_qc, contrast_rows

    try:
        for p, lab in [(bold, "bold"), (mask, "mask"), (events_stim, "events_stim"), (events_long, "events_long"), (confounds_path, "confounds")]:
            if not p.exists():
                raise FileNotFoundError(f"Missing {lab}: {p}")

        n_vols = get_nvols(bold)
        tr = get_tr_from_json_or_default(bold, default_tr=args.default_tr)
        run_qc["n_vols"] = int(n_vols)
        run_qc["tr"] = float(tr)

        events_glm, events_qc = build_glm_events(
            events_stim_path=events_stim,
            events_long_path=events_long,
            include_rating=not args.no_rating,
            include_temperature_pmod=not args.no_temp_pmod,
            pmod_column=args.temperature_pmod_column,
        )
        save_tsv(events_glm, run_dir / "events_glm.tsv")
        run_qc.update({f"events_{k}": v for k, v in events_qc.items() if isinstance(v, (str, int, float, bool, list))})

        confounds, conf_qc = load_and_clean_confounds(confounds_path)
        if confounds.shape[0] != n_vols:
            raise ValueError(f"Confounds rows ({confounds.shape[0]}) != n_vols ({n_vols})")
        save_tsv(confounds, run_dir / "confounds_used.tsv")
        run_qc.update({f"conf_{k}": v for k, v in conf_qc.items() if not isinstance(v, list)})

        high_pass_hz = 1.0 / float(args.high_pass_sec) if args.high_pass_sec and args.high_pass_sec > 0 else None
        signal_scaling = False if str(args.signal_scaling).lower() in {"false", "none", "no"} else int(args.signal_scaling)

        warnings_list: List[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            model = FirstLevelModel(
                t_r=tr,
                slice_time_ref=args.slice_time_ref,
                hrf_model=args.hrf_model,
                drift_model=args.drift_model,
                high_pass=high_pass_hz,
                noise_model=args.noise_model,
                mask_img=str(mask),
                smoothing_fwhm=None,
                signal_scaling=signal_scaling,
                standardize=False,
                minimize_memory=False,
                verbose=0,
            )

            model = model.fit(
                run_imgs=str(bold),
                events=events_glm,
                confounds=confounds,
            )

            warnings_list = [str(w.message) for w in caught]

        (run_dir / "warnings.txt").write_text("\n".join(warnings_list), encoding="utf-8")

        design = model.design_matrices_[0]
        save_tsv(design, run_dir / "design_matrix.tsv")
        plot_design_matrix_safe(design, run_dir / "design_matrix.png")
        plot_design_corr_safe(design, run_dir / "design_corr.png")

        design_rank, design_cond = matrix_rank_and_condition(design.to_numpy(dtype=float))
        df_resid = int(design.shape[0] - design_rank)
        run_qc.update({
            "design_n_rows": int(design.shape[0]),
            "design_n_cols": int(design.shape[1]),
            "design_rank": int(design_rank),
            "design_condition_number": float(design_cond) if np.isfinite(design_cond) else np.nan,
            "df_resid_approx": int(df_resid),
            "warnings_n": int(len(warnings_list)),
            "design_columns": list(design.columns),
        })

        # Design status flags.
        design_flags = []
        if df_resid < args.min_df_resid:
            design_flags.append("LOW_DF_RESID")
        if np.isfinite(design_cond) and design_cond > args.max_design_condition:
            design_flags.append("HIGH_DESIGN_CONDITION")
        if design_rank < design.shape[1]:
            design_flags.append("RANK_DEFICIENT_OR_COLLINEAR")
        run_qc["design_flags"] = ",".join(design_flags) if design_flags else "OK"

        targets = target_contrasts_from_design(list(design.columns))
        run_qc["contrasts_requested"] = targets
        run_qc["n_contrasts_requested"] = len(targets)

        for contrast in targets:
            cres = compute_and_save_contrast(model, contrast, run_dir)
            row_out = {
                "subject": subject,
                "run": run,
                "contrast": contrast,
                "ok": cres["ok"],
                "error": cres.get("error", ""),
                "beta_map": cres.get("maps", {}).get("beta", ""),
                "var_map": cres.get("maps", {}).get("var", ""),
                "stat_map": cres.get("maps", {}).get("stat", ""),
                "z_map": cres.get("maps", {}).get("z", ""),
            }
            contrast_rows.append(row_out)

        run_qc["n_contrasts_ok"] = int(sum(1 for r in contrast_rows if r["ok"]))
        run_qc["status"] = "OK"

        save_json(run_qc, run_dir / "run_qc.json")
        sentinel.write_text("OK\n", encoding="utf-8")

    except Exception as e:
        run_qc["status"] = "FAILED"
        run_qc["error"] = repr(e)
        save_json(run_qc, run_dir / "run_qc.json")
        (run_dir / "failure.txt").write_text(repr(e), encoding="utf-8")

    return run_qc, contrast_rows


# =============================================================================
# Fixed effects
# =============================================================================

def load_map(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img


def save_like(ref: nib.Nifti1Image, data: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    out = nib.Nifti1Image(data.astype(np.float32), ref.affine, ref.header.copy())
    out.to_filename(str(path))


def fixed_effects_maps(beta_paths: List[Path], var_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    betas = []
    vars_ = []
    ref = None

    for b, v in zip(beta_paths, var_paths):
        bd, img = load_map(b)
        vd, _ = load_map(v)
        if ref is None:
            ref = img
        betas.append(bd)
        vars_.append(vd)

    B = np.stack(betas, axis=0)
    V = np.stack(vars_, axis=0)
    V = np.where((V <= 0) | (~np.isfinite(V)), np.nan, V)
    W = 1.0 / V
    Wsum = np.nansum(W, axis=0)
    beta_fe = np.nansum(W * B, axis=0) / np.where(Wsum == 0, np.nan, Wsum)
    var_fe = 1.0 / np.where(Wsum == 0, np.nan, Wsum)
    beta_fe = np.nan_to_num(beta_fe, nan=0.0, posinf=0.0, neginf=0.0)
    var_fe = np.nan_to_num(var_fe, nan=np.inf, posinf=np.inf, neginf=np.inf)
    return beta_fe, var_fe, ref


def save_effect_var_stat_z(ref: nib.Nifti1Image, beta: np.ndarray, var: np.ndarray, stem: Path) -> Dict[str, str]:
    eps = 1e-12
    se = np.sqrt(np.maximum(var, eps))
    stat = beta / se
    z = stat  # For fixed-effects maps, this is an approximate z/stat map.

    paths = {
        "beta_map": str(stem.with_name(f"beta_{stem.name}.nii.gz")),
        "var_map": str(stem.with_name(f"var_{stem.name}.nii.gz")),
        "stat_map": str(stem.with_name(f"stat_{stem.name}.nii.gz")),
        "z_map": str(stem.with_name(f"z_{stem.name}.nii.gz")),
    }
    save_like(ref, beta, Path(paths["beta_map"]))
    save_like(ref, var, Path(paths["var_map"]))
    save_like(ref, stat, Path(paths["stat_map"]))
    save_like(ref, z, Path(paths["z_map"]))
    return paths


def build_subject_fixed_effects(
    contrast_inv: pd.DataFrame,
    outdir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    subjects = sorted(contrast_inv["subject"].dropna().unique().tolist())

    condition_targets = ["stim_passive", "stim_up", "stim_down", "stim_passive_temp", "stim_up_temp", "stim_down_temp"]

    for sub in subjects:
        sdir = outdir / sub / "subject_fixed_effects"
        ensure_dir(sdir)
        subdf = contrast_inv[(contrast_inv["subject"] == sub) & (contrast_inv["ok"] == True)].copy()

        fe_maps: Dict[str, Dict[str, str]] = {}
        fe_data: Dict[str, Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]] = {}

        # Fixed effects for each available run-level effect.
        for con in condition_targets:
            d = subdf[(subdf["contrast"] == con) & (subdf["beta_map"].astype(str) != "") & (subdf["var_map"].astype(str) != "")]
            if d.empty:
                continue
            beta_paths = [Path(p) for p in d["beta_map"].tolist()]
            var_paths = [Path(p) for p in d["var_map"].tolist()]
            try:
                beta, var, ref = fixed_effects_maps(beta_paths, var_paths)
                stem = sdir / f"{con}_FE"
                maps = save_effect_var_stat_z(ref, beta, var, stem)
                fe_maps[con] = maps
                fe_data[con] = (beta, var, ref)
                rows.append({
                    "subject": sub,
                    "effect": f"{con}_FE",
                    "kind": "condition_or_pmod_FE",
                    "n_run_maps": int(len(d)),
                    **maps,
                })
            except Exception as e:
                rows.append({
                    "subject": sub,
                    "effect": f"{con}_FE",
                    "kind": "condition_or_pmod_FE",
                    "n_run_maps": int(len(d)),
                    "error": repr(e),
                })

        # Subject-level contrasts from fixed effects.
        def add_contrast(label: str, terms: List[Tuple[str, float]]):
            available = [name in fe_data for name, _ in terms]
            if not all(available):
                return
            ref = fe_data[terms[0][0]][2]
            beta = None
            var = None
            for name, weight in terms:
                b, v, _ = fe_data[name]
                if beta is None:
                    beta = weight * b
                    var = (weight ** 2) * v
                else:
                    beta = beta + weight * b
                    var = var + (weight ** 2) * v
            maps = save_effect_var_stat_z(ref, beta, var, sdir / label)
            rows.append({
                "subject": sub,
                "effect": label,
                "kind": "subject_contrast_FE",
                "n_run_maps": np.nan,
                **maps,
            })

        add_contrast("up_minus_down_FE", [("stim_up", 1.0), ("stim_down", -1.0)])
        add_contrast("up_minus_passive_FE", [("stim_up", 1.0), ("stim_passive", -1.0)])
        add_contrast("down_minus_passive_FE", [("stim_down", 1.0), ("stim_passive", -1.0)])
        add_contrast("regulation_mean_minus_passive_FE", [("stim_up", 0.5), ("stim_down", 0.5), ("stim_passive", -1.0)])

    return pd.DataFrame(rows)


def write_group_inputs(subject_fx: pd.DataFrame, outdir: Path) -> None:
    gdir = outdir / "group_inputs"
    ensure_dir(gdir)

    if subject_fx.empty:
        return

    save_tsv(subject_fx, gdir / "group_inputs_all_effects.tsv")

    for effect, d in subject_fx.groupby("effect"):
        d2 = d[d["beta_map"].astype(str) != ""].copy()
        if d2.empty:
            continue
        safe = str(effect).replace("/", "_").replace(" ", "_")
        cols = ["subject", "effect", "beta_map", "var_map", "stat_map", "z_map", "kind", "n_run_maps"]
        cols = [c for c in cols if c in d2.columns]
        save_tsv(d2[cols].sort_values("subject"), gdir / f"group_input_{safe}.tsv")


# =============================================================================
# Report
# =============================================================================

def write_report(outdir: Path, manifest: Dict[str, object], run_inv: pd.DataFrame, contrast_inv: pd.DataFrame, subject_fx: pd.DataFrame) -> None:
    def col_numeric(name: str) -> pd.Series:
        if name in run_inv.columns:
            return pd.to_numeric(run_inv[name], errors="coerce")
        return pd.Series(dtype=float)

    def n_status(values: Sequence[str]) -> int:
        if "status" not in run_inv.columns:
            return 0
        return int(run_inv["status"].isin(list(values)).sum())

    n_runs = len(run_inv)
    n_ok = n_status(["OK"])
    n_skipped = n_status(["SKIPPED_DONE"])
    n_completed = n_status(["OK", "SKIPPED_DONE"])
    n_failed = n_status(["FAILED"])

    lines = []
    lines.append("# First-level GLM report\n\n")
    lines.append("## Overview\n")
    lines.append(f"- Generated: {manifest['timestamp_local']}\n")
    lines.append(f"- Input inventory: `{manifest['input_inventory']}`\n")
    lines.append(f"- Output directory: `{manifest['outdir']}`\n")
    lines.append(f"- Runs input: {manifest['n_runs_input']}\n")
    lines.append(f"- Runs fitted OK in this execution: {n_ok}\n")
    lines.append(f"- Runs resumed from existing outputs: {n_skipped}\n")
    lines.append(f"- Runs completed total: {n_completed}\n")
    lines.append(f"- Runs failed: {n_failed}\n\n")

    lines.append("## Model specification\n")
    lines.append(f"- HRF model: `{manifest['hrf_model']}`\n")
    lines.append(f"- Drift model: `{manifest['drift_model']}`\n")
    lines.append(f"- High-pass cutoff: {manifest['high_pass_sec']} s\n")
    lines.append(f"- Noise model: `{manifest['noise_model']}`\n")
    lines.append(f"- Signal scaling: `{manifest['signal_scaling']}`\n")
    lines.append(f"- Rating nuisance included: {manifest['include_rating']}\n")
    lines.append(f"- Temperature pmod included: {manifest['include_temperature_pmod']}\n")
    lines.append(f"- Temperature pmod column: `{manifest['temperature_pmod_column']}`\n\n")

    lines.append("## Design QC\n")
    df_resid = col_numeric("df_resid_approx").dropna()
    design_flags = run_inv["design_flags"] if "design_flags" in run_inv.columns else pd.Series([], dtype=str)
    if len(df_resid):
        lines.append(f"- Median approximate residual DOF: {df_resid.median():.1f}\n")
        lines.append(f"- Minimum approximate residual DOF: {df_resid.min():.1f}\n")
    else:
        lines.append("- Residual DOF summary unavailable; inspect run-level failures or skipped outputs.\n")
    if len(design_flags):
        lines.append(f"- Runs with design_flags != OK: {int((design_flags.astype(str) != 'OK').sum())}\n")
        flag_counts = design_flags.astype(str).value_counts().to_dict()
        for k, v in sorted(flag_counts.items()):
            lines.append(f"  - {k}: {v}\n")
    lines.append("\n")

    lines.append("## Run-level contrasts\n")
    lines.append(f"- Run contrast rows: {len(contrast_inv)}\n")
    if len(contrast_inv) and "contrast" in contrast_inv.columns:
        ok_counts = contrast_inv.groupby("contrast")["ok"].sum().to_dict()
        for k, v in sorted(ok_counts.items()):
            lines.append(f"- {k}: {int(v)} run maps\n")
    lines.append("\n")

    lines.append("## Subject fixed effects\n")
    lines.append(f"- Subject fixed-effect rows: {len(subject_fx)}\n")
    if len(subject_fx) and "effect" in subject_fx.columns:
        for effect, d in subject_fx.groupby("effect"):
            lines.append(f"- {effect}: {d['subject'].nunique()} subjects\n")
    lines.append("\n")

    lines.append("## Motion / sensitivity strategy\n")
    lines.append("- This script uses the pre-GLM confounds prepared with the fd05 spike policy.\n")
    lines.append("- The GLM does not re-run QC exclusions. QC/sensitivity labels are preserved in the run inventory.\n")
    lines.append("- Group-level robustness is assessed using the sensitivity sets prepared before GLM.\n\n")

    lines.append("## Key outputs\n")
    lines.append("- `tables/first_level_run_inventory.tsv`\n")
    lines.append("- `tables/run_contrast_inventory.tsv`\n")
    lines.append("- `tables/subject_fixed_effects_inventory.tsv`\n")
    lines.append("- `group_inputs/group_input_*.tsv`\n")
    lines.append("- `sub-XX/run-YY/run_qc.json`\n")
    lines.append("- `sub-XX/run-YY/design_matrix.tsv`\n")
    lines.append("- `sub-XX/run-YY/beta_*.nii.gz`, `var_*.nii.gz`, `stat_*.nii.gz`, `z_*.nii.gz`\n")

    (outdir / "first_level_glm_report.md").write_text("".join(lines), encoding="utf-8")



# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="analysis-level first-level GLM for ds000140.")
    parser.add_argument("--inventory", required=True, type=str, help="Path to first_level_inputs_analysis/tables/first_level_run_inventory.tsv")
    parser.add_argument("--outdir", required=True, type=str, help="Output directory for first-level GLM")
    parser.add_argument("--include-set", default="ready", choices=["ready", "all"], help="Rows from inventory to fit")
    parser.add_argument("--hrf-model", default="spm + derivative", help="Nilearn HRF model")
    parser.add_argument("--drift-model", default="cosine", help="Nilearn drift model")
    parser.add_argument("--high-pass-sec", default=180.0, type=float, help="High-pass cutoff in seconds")
    parser.add_argument("--noise-model", default="ar1", help="Nilearn noise model")
    parser.add_argument("--slice-time-ref", default=0.5, type=float)
    parser.add_argument("--signal-scaling", default="0", help="Nilearn signal_scaling. Use 0 or false.")
    parser.add_argument("--default-tr", default=2.0, type=float)
    parser.add_argument("--temperature-pmod-column", default="temperature_z_subject")
    parser.add_argument("--no-temp-pmod", action="store_true", help="Do not include temperature pmods")
    parser.add_argument("--no-rating", action="store_true", help="Do not include rating-period nuisance")
    parser.add_argument("--min-df-resid", default=50, type=int)
    parser.add_argument("--max-design-condition", default=1e10, type=float)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inventory_path = Path(args.inventory).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    tables = outdir / "tables"
    ensure_dir(outdir)
    ensure_dir(tables)

    inv = load_firstlevel_inventory(inventory_path, include_set=args.include_set)

    run_rows: List[Dict[str, object]] = []
    contrast_rows: List[Dict[str, object]] = []

    for i, (_, row) in enumerate(inv.iterrows(), start=1):
        print(f"[{i}/{len(inv)}] Fitting {row['subject']} run-{int(row['run']):02d}", flush=True)
        rq, cr = fit_one_run(row, outdir, args)
        run_rows.append(rq)
        contrast_rows.extend(cr)

    run_inv = pd.DataFrame(run_rows)
    if not run_inv.empty:
        run_inv = run_inv.sort_values(["subject", "run"]).reset_index(drop=True)
    contrast_inv = pd.DataFrame(contrast_rows)
    if not contrast_inv.empty:
        contrast_inv = contrast_inv.sort_values(["subject", "run", "contrast"]).reset_index(drop=True)

    save_tsv(run_inv, tables / "first_level_run_inventory.tsv")
    save_tsv(contrast_inv, tables / "run_contrast_inventory.tsv")
    save_tsv(run_inv[run_inv["status"] == "FAILED"].copy(), tables / "first_level_failed.tsv")

    # Subject fixed effects from run contrast inventory.
    if not contrast_inv.empty:
        subject_fx = build_subject_fixed_effects(contrast_inv, outdir)
    else:
        subject_fx = pd.DataFrame()
    save_tsv(subject_fx, tables / "subject_fixed_effects_inventory.tsv")
    write_group_inputs(subject_fx, outdir)

    manifest = {
        "timestamp_local": now_local_iso(),
        "script": Path(__file__).name,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "input_inventory": str(inventory_path),
        "input_inventory_sha256": sha256_file(inventory_path),
        "outdir": str(outdir),
        "n_runs_input": int(inv.shape[0]),
        "include_set": args.include_set,
        "hrf_model": args.hrf_model,
        "drift_model": args.drift_model,
        "high_pass_sec": float(args.high_pass_sec),
        "noise_model": args.noise_model,
        "slice_time_ref": float(args.slice_time_ref),
        "signal_scaling": args.signal_scaling,
        "include_rating": not args.no_rating,
        "include_temperature_pmod": not args.no_temp_pmod,
        "temperature_pmod_column": args.temperature_pmod_column,
        "min_df_resid": int(args.min_df_resid),
        "max_design_condition": float(args.max_design_condition),
    }
    save_json(manifest, outdir / "manifest.json")
    write_report(outdir, manifest, run_inv, contrast_inv, subject_fx)

    print("Processing completed successfully.")
    if len(run_inv) and "status" in run_inv.columns:
        print(f"Runs fitted OK: {int((run_inv['status'] == 'OK').sum())}")
        print(f"Runs resumed/skipped: {int((run_inv['status'] == 'SKIPPED_DONE').sum())}")
        print(f"Runs failed: {int((run_inv['status'] == 'FAILED').sum())}")
    else:
        print("Runs fitted OK: 0")
        print("Runs resumed/skipped: 0")
        print("Runs failed: 0")
    print(f"Output directory: {outdir}")
    print(f"Run inventory: {tables / 'first_level_run_inventory.tsv'}")


if __name__ == "__main__":
    main()
