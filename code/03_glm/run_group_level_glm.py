#!/usr/bin/env python3
"""
run_group_level_glm.py

Group-level GLM and motion-robustness analysis for ds000140.

Rationale
---------
A strict run-exclusion set is highly conservative for regulation contrasts
because ds000140 contains only two regulation runs per participant. Excluding
all run-level sensitivity candidates can markedly reduce the available sample.

The implemented analysis sets are:
  1) main:
       all first-level runs that passed severe QC and GLM checks.
  2) fd_bad_run_exclude:
       rebuild subject fixed effects after excluding only FD-based bad/severe
       runs, not all nuisance-burden candidates.
  3) high_motion_subject_exclude:
       use main subject effects but exclude high-motion participants.
  4) motion_covariate_mean_fd:
       use main subject effects and add participant mean FD as a group-level covariate.
  5) strict:
       an optional stress test excluding all sensitivity candidates; it is not
       used as the primary robustness criterion for regulation effects.

Default effects
---------------
  up_minus_down_FE
  up_minus_passive_FE
  down_minus_passive_FE
  regulation_mean_minus_passive_FE
  stim_passive_FE

Required inputs
---------------
  --glm-dir firstlevel_GLM_analysis
  --sensitivity-sets first_level_inputs_analysis/tables/sensitivity_sets.tsv
  --outdir group_level_analysis

Optional
--------
  --qc-subject-summary qc_fmriprep_analysis/tables/subject_qc_summary.tsv
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
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.image import load_img


# =============================================================================
# Utilities
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


def parse_list(x: str) -> List[str]:
    return [p.strip() for p in str(x).split(",") if p.strip()]


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


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def zscore(x: pd.Series) -> pd.Series:
    x = safe_num(x)
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd < 1e-12:
        return x * 0.0
    return (x - x.mean()) / sd


def strip_nii_gz(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def load_map(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img


def save_like(ref: nib.Nifti1Image, data: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    out = nib.Nifti1Image(data.astype(np.float32), ref.affine, ref.header.copy())
    out.to_filename(str(path))


def save_beta_var_stat_z(ref: nib.Nifti1Image, beta: np.ndarray, var: np.ndarray, stem: Path) -> Dict[str, str]:
    eps = 1e-12
    var2 = np.where((var <= 0) | (~np.isfinite(var)), np.nan, var)
    stat = beta / np.sqrt(np.maximum(var2, eps))
    stat = np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)
    z = stat.copy()

    paths = {
        "beta_map": str(stem.with_name(f"beta_{stem.name}.nii.gz")),
        "var_map": str(stem.with_name(f"var_{stem.name}.nii.gz")),
        "stat_map": str(stem.with_name(f"stat_{stem.name}.nii.gz")),
        "z_map": str(stem.with_name(f"z_{stem.name}.nii.gz")),
    }
    save_like(ref, beta, Path(paths["beta_map"]))
    save_like(ref, np.nan_to_num(var2, nan=np.inf, posinf=np.inf, neginf=np.inf), Path(paths["var_map"]))
    save_like(ref, stat, Path(paths["stat_map"]))
    save_like(ref, z, Path(paths["z_map"]))
    return paths


def fixed_effects(beta_paths: List[Path], var_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    betas, vars_ = [], []
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
    beta = np.nansum(W * B, axis=0) / np.where(Wsum == 0, np.nan, Wsum)
    var = 1.0 / np.where(Wsum == 0, np.nan, Wsum)
    beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
    var = np.nan_to_num(var, nan=np.inf, posinf=np.inf, neginf=np.inf)
    return beta, var, ref


def image_corr(path_a: str, path_b: str) -> float:
    if not path_a or not path_b:
        return np.nan
    pa, pb = Path(path_a), Path(path_b)
    if not pa.exists() or not pb.exists():
        return np.nan
    a = nib.load(str(pa)).get_fdata(dtype=np.float32)
    b = nib.load(str(pb)).get_fdata(dtype=np.float32)
    mask = np.isfinite(a) & np.isfinite(b) & ((a != 0) | (b != 0))
    if mask.sum() < 10:
        return np.nan
    av = a[mask].ravel()
    bv = b[mask].ravel()
    if av.std() == 0 or bv.std() == 0:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])


def map_summary(path: str) -> Dict[str, float]:
    if not path or not Path(path).exists():
        return {"max": np.nan, "min": np.nan, "max_abs": np.nan}
    data = nib.load(path).get_fdata(dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {"max": np.nan, "min": np.nan, "max_abs": np.nan}
    return {
        "max": float(np.max(finite)),
        "min": float(np.min(finite)),
        "max_abs": float(np.max(np.abs(finite))),
    }


# =============================================================================
# Load inputs
# =============================================================================

def load_run_contrasts(glm_dir: Path) -> pd.DataFrame:
    p = glm_dir / "tables" / "run_contrast_inventory.tsv"
    if p.exists():
        df = pd.read_csv(p, sep="\t")
        df["subject"] = df["subject"].map(clean_subject)
        df["run"] = pd.to_numeric(df["run"], errors="coerce").astype("Int64")
        df["ok"] = df["ok"].map(bool_from_any)
        return df

    # Search alternative locations when the primary path is unavailable.
    rows = []
    for run_dir in sorted(glm_dir.glob("sub-*/run-*")):
        sub = run_dir.parent.name
        run = int(run_dir.name.split("-")[-1])
        for beta in sorted(run_dir.glob("beta_*.nii.gz")):
            contrast = strip_nii_gz(beta).replace("beta_", "", 1)
            var = run_dir / f"var_{contrast}.nii.gz"
            stat = run_dir / f"stat_{contrast}.nii.gz"
            z = run_dir / f"z_{contrast}.nii.gz"
            rows.append({
                "subject": sub,
                "run": run,
                "contrast": contrast,
                "ok": beta.exists(),
                "beta_map": str(beta),
                "var_map": str(var) if var.exists() else "",
                "stat_map": str(stat) if stat.exists() else "",
                "z_map": str(z) if z.exists() else "",
            })
    return pd.DataFrame(rows)


def load_run_inventory(glm_dir: Path) -> pd.DataFrame:
    p = glm_dir / "tables" / "first_level_run_inventory.tsv"
    if not p.exists():
        raise FileNotFoundError(f"Missing first-level run inventory: {p}")
    df = pd.read_csv(p, sep="\t")
    df["subject"] = df["subject"].map(clean_subject)
    df["run"] = pd.to_numeric(df["run"], errors="coerce").astype("Int64")
    for c in ["fd_mean", "fd_max", "pct_fd_gt_0p5", "motion_outlier_volumes_pct"]:
        if c in df.columns:
            df[c] = safe_num(df[c])
    return df


def load_sensitivity_sets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["subject"] = df["subject"].map(clean_subject)
    df["run"] = pd.to_numeric(df["run"], errors="coerce").astype("Int64")
    for c in ["main_include", "strict_sensitivity_include", "sensitivity_exclude_candidate"]:
        if c in df.columns:
            df[c] = df[c].map(bool_from_any)
    return df


def load_qc_subject_summary(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    df["subject"] = df["subject"].map(clean_subject)
    for c in ["high_motion_subject_candidate", "high_nuisance_subject_candidate"]:
        if c in df.columns:
            df[c] = df[c].map(bool_from_any)
    return df


# =============================================================================
# Subject motion summary and allowed runs
# =============================================================================

def derive_subject_motion_summary(run_inv: pd.DataFrame, reg_runs: Sequence[int] = (3, 7)) -> pd.DataFrame:
    d = run_inv.copy()
    d["motion_bad_or_severe"] = d["motion_flag"].astype(str).str.contains("BAD|SEVERE", regex=True)
    rows = []
    for sub, ds in d.groupby("subject"):
        reg = ds[ds["run"].isin(reg_runs)]
        rows.append({
            "subject": sub,
            "mean_fd_mean": float(ds["fd_mean"].mean()),
            "median_fd_mean": float(ds["fd_mean"].median()),
            "max_fd_mean": float(ds["fd_mean"].max()),
            "max_fd_max": float(ds["fd_max"].max()),
            "mean_pct_fd_gt_0p5": float(ds["pct_fd_gt_0p5"].mean()),
            "max_pct_fd_gt_0p5": float(ds["pct_fd_gt_0p5"].max()),
            "n_motion_bad_or_severe_runs": int(ds["motion_bad_or_severe"].sum()),
            "n_reg_motion_bad_or_severe_runs": int(reg["motion_bad_or_severe"].sum()) if len(reg) else 0,
            "mean_reg_fd_mean": float(reg["fd_mean"].mean()) if len(reg) else np.nan,
            "max_reg_fd_mean": float(reg["fd_mean"].max()) if len(reg) else np.nan,
            "max_reg_pct_fd_gt_0p5": float(reg["pct_fd_gt_0p5"].max()) if len(reg) else np.nan,
        })
    out = pd.DataFrame(rows)
    return out


def identify_high_motion_subjects(
    subject_motion: pd.DataFrame,
    qc_subject_summary: Optional[pd.DataFrame],
    mean_fd_threshold: float,
    min_bad_runs: int,
    exclude_qc_high_motion: bool = True,
) -> Tuple[List[str], pd.DataFrame]:
    sm = subject_motion.copy()
    sm["high_motion_derived"] = (
        (sm["mean_fd_mean"] > mean_fd_threshold)
        | (sm["n_motion_bad_or_severe_runs"] >= min_bad_runs)
        | (sm["n_reg_motion_bad_or_severe_runs"] >= 1)
    )

    if qc_subject_summary is not None and "high_motion_subject_candidate" in qc_subject_summary.columns and exclude_qc_high_motion:
        q = qc_subject_summary[["subject", "high_motion_subject_candidate"]].copy()
        sm = sm.merge(q, on="subject", how="left")
        sm["high_motion_subject_candidate"] = sm["high_motion_subject_candidate"].fillna(False).astype(bool)
    else:
        sm["high_motion_subject_candidate"] = False

    sm["exclude_high_motion_subject"] = sm["high_motion_derived"] | sm["high_motion_subject_candidate"]
    subs = sorted(sm.loc[sm["exclude_high_motion_subject"], "subject"].tolist())
    return subs, sm


def build_allowed_runs(
    run_inv: pd.DataFrame,
    sensitivity_sets: pd.DataFrame,
    set_name: str,
    high_motion_subjects: Sequence[str],
    fd_bad_mean: float,
    fd_bad_pct: float,
) -> pd.DataFrame:
    set_name = set_name.lower().strip()
    main = run_inv[run_inv["status"].astype(str).isin(["OK", "SKIPPED_DONE"])].copy()

    if set_name == "main":
        return main[["subject", "run"]].drop_duplicates()

    if set_name in {"strict", "strict_sensitivity"}:
        if "strict_sensitivity_include" not in sensitivity_sets.columns:
            raise ValueError("sensitivity_sets missing strict_sensitivity_include")
        return sensitivity_sets[sensitivity_sets["strict_sensitivity_include"]][["subject", "run"]].drop_duplicates()

    if set_name == "fd_bad_run_exclude":
        d = main.copy()
        motion_bad = d["motion_flag"].astype(str).str.contains("BAD|SEVERE", regex=True)
        fd_bad = (d["fd_mean"] > fd_bad_mean) | (d["pct_fd_gt_0p5"] > fd_bad_pct)
        d = d[~(motion_bad | fd_bad)].copy()
        return d[["subject", "run"]].drop_duplicates()

    if set_name == "high_motion_subject_exclude":
        d = main[~main["subject"].isin(high_motion_subjects)].copy()
        return d[["subject", "run"]].drop_duplicates()

    if set_name.startswith("motion_covariate"):
        return main[["subject", "run"]].drop_duplicates()

    raise ValueError(f"Unknown analysis set: {set_name}")


# =============================================================================
# Subject fixed effects
# =============================================================================

def compute_subject_condition_fe(
    run_contrasts: pd.DataFrame,
    allowed: pd.DataFrame,
    outdir: Path,
    set_name: str,
) -> pd.DataFrame:
    rc = run_contrasts.copy()
    rc = rc[rc["ok"] == True].copy()
    allowed_idx = set(zip(allowed["subject"], allowed["run"].astype(int)))
    rc = rc[rc.apply(lambda r: (r["subject"], int(r["run"])) in allowed_idx, axis=1)].copy()

    rows = []
    base_effects = ["stim_passive", "stim_up", "stim_down", "stim_passive_temp", "stim_up_temp", "stim_down_temp"]

    for sub in sorted(rc["subject"].unique().tolist()):
        sdir = outdir / "subject_effects" / set_name / sub
        ensure_dir(sdir)
        sdata = {}

        for eff in base_effects:
            d = rc[(rc["subject"] == sub) & (rc["contrast"] == eff)].copy()
            d = d[(d["beta_map"].astype(str) != "") & (d["var_map"].astype(str) != "")]
            if d.empty:
                continue
            beta_paths = [Path(p) for p in d["beta_map"].tolist()]
            var_paths = [Path(p) for p in d["var_map"].tolist()]
            if not all(p.exists() for p in beta_paths + var_paths):
                continue
            try:
                beta, var, ref = fixed_effects(beta_paths, var_paths)
                maps = save_beta_var_stat_z(ref, beta, var, sdir / f"{eff}_FE")
                sdata[eff] = (beta, var, ref)
                rows.append({
                    "set_name": set_name,
                    "subject": sub,
                    "effect": f"{eff}_FE",
                    "effect_family": "condition_or_pmod",
                    "n_run_maps": int(len(d)),
                    **maps,
                })
            except Exception as e:
                rows.append({
                    "set_name": set_name,
                    "subject": sub,
                    "effect": f"{eff}_FE",
                    "effect_family": "condition_or_pmod",
                    "n_run_maps": int(len(d)),
                    "error": repr(e),
                })

        def add_contrast(label: str, terms: List[Tuple[str, float]]):
            if not all(name in sdata for name, _ in terms):
                return
            ref = sdata[terms[0][0]][2]
            beta = None
            var = None
            n_terms = []
            for name, weight in terms:
                b, v, _ = sdata[name]
                if beta is None:
                    beta = weight * b
                    var = (weight ** 2) * v
                else:
                    beta = beta + weight * b
                    var = var + (weight ** 2) * v
                n_terms.append(name)
            maps = save_beta_var_stat_z(ref, beta, var, sdir / label)
            rows.append({
                "set_name": set_name,
                "subject": sub,
                "effect": label,
                "effect_family": "subject_contrast",
                "n_run_maps": np.nan,
                "contrast_terms": ";".join(n_terms),
                **maps,
            })

        add_contrast("up_minus_down_FE", [("stim_up", 1.0), ("stim_down", -1.0)])
        add_contrast("up_minus_passive_FE", [("stim_up", 1.0), ("stim_passive", -1.0)])
        add_contrast("down_minus_passive_FE", [("stim_down", 1.0), ("stim_passive", -1.0)])
        add_contrast("regulation_mean_minus_passive_FE", [("stim_up", 0.5), ("stim_down", 0.5), ("stim_passive", -1.0)])

    return pd.DataFrame(rows)


# =============================================================================
# Group-level GLM
# =============================================================================

def make_group_design(subjects: List[str], set_name: str, subject_motion: pd.DataFrame, covariate: Optional[str]) -> pd.DataFrame:
    design = pd.DataFrame({"subject": subjects, "intercept": np.ones(len(subjects))})
    if covariate:
        cov = subject_motion[["subject", covariate]].copy()
        design = design.merge(cov, on="subject", how="left")
        design[covariate] = zscore(design[covariate])
        design = design.rename(columns={covariate: f"{covariate}_z"})
    return design


def threshold_maps(z_path: str, outdir: Path, effect: str, alpha_fdr: float = 0.05) -> Dict[str, object]:
    outputs = {}
    if not z_path or not Path(z_path).exists():
        return outputs

    z_img = load_img(z_path)

    # FDR threshold. If no voxel survives, nilearn may return inf threshold.
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            fdr_img, fdr_thr = threshold_stats_img(z_img, alpha=alpha_fdr, height_control="fdr", cluster_threshold=0)
        outputs["fdr_threshold"] = float(fdr_thr) if np.isfinite(fdr_thr) else np.inf
        if np.isfinite(outputs["fdr_threshold"]):
            fdr_path = outdir / f"group_z_{effect}_fdr{str(alpha_fdr).replace('.', 'p')}.nii.gz"
            fdr_img.to_filename(str(fdr_path))
            outputs["fdr_map"] = str(fdr_path)
        else:
            outputs["fdr_map"] = ""
            outputs["fdr_note"] = "No voxels survived FDR threshold; threshold is infinite."
    except Exception as e:
        outputs["fdr_map"] = ""
        outputs["fdr_error"] = repr(e)
        outputs["fdr_threshold"] = np.nan

    # Uncorrected p<0.001 two-sided z approx 3.29. Use fixed explicit threshold.
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            unc_img, unc_thr = threshold_stats_img(z_img, threshold=3.29, height_control=None, cluster_threshold=0)
        unc_path = outdir / f"group_z_{effect}_unc001_z3p29.nii.gz"
        unc_img.to_filename(str(unc_path))
        outputs["unc001_map"] = str(unc_path)
        outputs["unc001_threshold"] = 3.29
    except Exception as e:
        outputs["unc001_map"] = ""
        outputs["unc001_error"] = repr(e)
        outputs["unc001_threshold"] = 3.29

    return outputs


def group_level_one_sample(
    effect_rows: pd.DataFrame,
    set_name: str,
    effect: str,
    outdir: Path,
    subject_motion: pd.DataFrame,
    covariate: Optional[str] = None,
) -> Dict[str, object]:
    ensure_dir(outdir)
    d = effect_rows.copy()
    d = d[d["beta_map"].astype(str) != ""].sort_values("subject")
    d = d[d["beta_map"].map(lambda p: Path(str(p)).exists())].copy()

    maps = [str(p) for p in d["beta_map"].tolist()]
    subjects = d["subject"].tolist()

    res = {
        "set_name": set_name,
        "effect": effect,
        "n_subjects": int(len(maps)),
        "covariate": covariate or "",
        "status": "UNKNOWN",
        "error": "",
    }

    if len(maps) < 3:
        res["status"] = "FAILED"
        res["error"] = f"Too few subjects: {len(maps)}"
        return res

    try:
        design = make_group_design(subjects, set_name, subject_motion, covariate=covariate)
        design_path = outdir / f"group_design_{effect}.tsv"
        save_tsv(design, design_path)

        # Exclude the subject identifier before model fitting.
        X = design.drop(columns=["subject"])

        model = SecondLevelModel(smoothing_fwhm=None, n_jobs=1)
        model = model.fit(maps, design_matrix=X)

        outputs = {"group_design": str(design_path)}
        for output_type, prefix in [
            ("effect_size", "group_beta"),
            ("effect_variance", "group_var"),
            ("stat", "group_stat"),
            ("z_score", "group_z"),
            ("p_value", "group_p"),
        ]:
            try:
                img = model.compute_contrast("intercept", output_type=output_type)
                path = outdir / f"{prefix}_{effect}.nii.gz"
                img.to_filename(str(path))
                outputs[f"{prefix}_map"] = str(path)
            except Exception as e:
                outputs[f"{prefix}_map"] = ""
                outputs[f"{prefix}_error"] = repr(e)

        # Save the covariate effect map when applicable.
        if covariate:
            cov_col = f"{covariate}_z"
            for output_type, prefix in [
                ("effect_size", "cov_beta"),
                ("z_score", "cov_z"),
                ("stat", "cov_stat"),
            ]:
                try:
                    img = model.compute_contrast(cov_col, output_type=output_type)
                    path = outdir / f"{prefix}_{cov_col}_{effect}.nii.gz"
                    img.to_filename(str(path))
                    outputs[f"{prefix}_map"] = str(path)
                except Exception as e:
                    outputs[f"{prefix}_map"] = ""
                    outputs[f"{prefix}_error"] = repr(e)

        # Threshold z map
        outputs.update(threshold_maps(outputs.get("group_z_map", ""), outdir, effect))

        # Summarize z-statistics.
        zsum = map_summary(outputs.get("group_z_map", ""))
        outputs["z_max"] = zsum["max"]
        outputs["z_min"] = zsum["min"]
        outputs["z_max_abs"] = zsum["max_abs"]

        save_tsv(pd.DataFrame({"subject": subjects, "map": maps}), outdir / f"group_subject_maps_{effect}.tsv")

        res.update(outputs)
        res["status"] = "OK"

    except Exception as e:
        res["status"] = "FAILED"
        res["error"] = repr(e)

    return res


def run_group_level(subject_effects: pd.DataFrame, effects: List[str], set_names: List[str], outdir: Path, subject_motion: pd.DataFrame, min_subjects: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    failures = []

    for set_name in set_names:
        covariate = None
        source_set = set_name
        if set_name.startswith("motion_covariate"):
            source_set = "main"
            covariate = "mean_fd_mean"

        for effect in effects:
            d = subject_effects[(subject_effects["set_name"] == source_set) & (subject_effects["effect"] == effect)].copy()
            eff_dir = outdir / "group_maps" / set_name / effect
            res = group_level_one_sample(d, set_name=set_name, effect=effect, outdir=eff_dir, subject_motion=subject_motion, covariate=covariate)
            res["source_subject_effect_set"] = source_set
            res["min_subjects_required"] = int(min_subjects)
            if res["n_subjects"] < min_subjects and res["status"] == "OK":
                res["status"] = "WARNING_LOW_N"
                res["warning"] = f"n_subjects={res['n_subjects']} < min_subjects={min_subjects}"
            results.append(res)
            if res["status"] == "FAILED":
                failures.append(res)

    return pd.DataFrame(results), pd.DataFrame(failures)


# =============================================================================
# Robustness summary
# =============================================================================

def robustness_summary(group_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if group_results.empty:
        return pd.DataFrame(rows)

    main = group_results[group_results["set_name"] == "main"].copy()
    for _, m in main.iterrows():
        eff = m["effect"]
        for _, r in group_results[(group_results["effect"] == eff) & (group_results["set_name"] != "main")].iterrows():
            rows.append({
                "effect": eff,
                "comparison_set": r["set_name"],
                "main_status": m["status"],
                "comparison_status": r["status"],
                "main_n_subjects": m["n_subjects"],
                "comparison_n_subjects": r["n_subjects"],
                "delta_n_subjects": r["n_subjects"] - m["n_subjects"],
                "corr_group_beta": image_corr(m.get("group_beta_map", ""), r.get("group_beta_map", "")),
                "corr_group_z": image_corr(m.get("group_z_map", ""), r.get("group_z_map", "")),
                "main_z_max_abs": m.get("z_max_abs", np.nan),
                "comparison_z_max_abs": r.get("z_max_abs", np.nan),
                "main_fdr_threshold": m.get("fdr_threshold", np.nan),
                "comparison_fdr_threshold": r.get("fdr_threshold", np.nan),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Report
# =============================================================================

def write_report(outdir: Path, manifest: Dict[str, object], subject_motion: pd.DataFrame, subject_effects: pd.DataFrame, group_results: pd.DataFrame, robust: pd.DataFrame) -> None:
    lines = []
    lines.append("# Group-level GLM report \n\n")
    lines.append("## Overview\n")
    lines.append(f"- Generated: {manifest['timestamp_local']}\n")
    lines.append(f"- GLM directory: `{manifest['glm_dir']}`\n")
    lines.append(f"- Sensitivity table: `{manifest['sensitivity_sets']}`\n")
    lines.append(f"- Output directory: `{manifest['outdir']}`\n")
    lines.append(f"- Sets: {', '.join(manifest['sets'])}\n")
    lines.append(f"- Effects: {', '.join(manifest['effects'])}\n\n")

    lines.append("## Robustness sets\n")
    lines.append("- `main`: all runs passing severe QC and first-level GLM.\n")
    lines.append("- `fd_bad_run_exclude`: rebuild subject effects excluding only FD-based bad/severe runs.\n")
    lines.append("- `high_motion_subject_exclude`: use main subject effects but exclude high-motion subjects.\n")
    lines.append("- `motion_covariate_mean_fd`: use main subject effects and add subject mean FD as a second-level covariate.\n")
    lines.append("- `strict`: optional stress test excluding all sensitivity candidates; can be underpowered for regulation contrasts.\n\n")

    if len(subject_motion):
        high = subject_motion[subject_motion.get("exclude_high_motion_subject", False)]
        lines.append("## High-motion subjects\n")
        lines.append(f"- High-motion subjects excluded in high_motion_subject_exclude: {len(high)}\n")
        if len(high):
            lines.append(f"- Subjects: {', '.join(high['subject'].tolist())}\n")
        lines.append("\n")

    lines.append("## Subject fixed effects\n")
    lines.append(f"- Subject-effect rows: {len(subject_effects)}\n")
    if len(subject_effects):
        for set_name, dset in subject_effects.groupby("set_name"):
            lines.append(f"- {set_name}: {dset['subject'].nunique()} subjects with at least one effect row\n")
    lines.append("\n")

    lines.append("## Group-level results\n")
    if len(group_results):
        for _, r in group_results.iterrows():
            lines.append(
                f"- {r['set_name']} | {r['effect']}: status={r['status']}, "
                f"n={r['n_subjects']}, zmaxabs={r.get('z_max_abs', np.nan):.3f}\n"
            )
    lines.append("\n")

    lines.append("## Robustness summary versus main\n")
    if len(robust):
        for _, r in robust.iterrows():
            lines.append(
                f"- {r['effect']} | {r['comparison_set']}: n={r['comparison_n_subjects']}, "
                f"beta corr={r['corr_group_beta']:.3f}, z corr={r['corr_group_z']:.3f}\n"
            )
    else:
        lines.append("- No robustness summary available.\n")
    lines.append("\n")

    lines.append("## Interpretation policy\n")
    lines.append("- The `main` analysis set defines primary inference.\n")
    lines.append("- Robustness is evaluated by similarity across FD-bad-run exclusion, high-motion-participant exclusion, and motion-covariate control.\n")
    lines.append("- The strict set is intentionally conservative and may be underpowered for regulation contrasts due to the two-run regulation design.\n")
    lines.append("- Inference is based on unthresholded group maps, corrected/thresholded maps, and consistency across robustness analyses.\n")

    (outdir / "group_level_report.md").write_text("".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glm-dir", required=True, type=str)
    ap.add_argument("--sensitivity-sets", required=True, type=str)
    ap.add_argument("--outdir", required=True, type=str)
    ap.add_argument("--qc-subject-summary", default="", type=str)
    ap.add_argument("--sets", default="main,fd_bad_run_exclude,high_motion_subject_exclude,motion_covariate_mean_fd", type=str)
    ap.add_argument("--effects", default="up_minus_down_FE,up_minus_passive_FE,down_minus_passive_FE,regulation_mean_minus_passive_FE,stim_passive_FE", type=str)
    ap.add_argument("--min-subjects", default=10, type=int)
    ap.add_argument("--fd-bad-mean", default=0.30, type=float)
    ap.add_argument("--fd-bad-pct", default=10.0, type=float)
    ap.add_argument("--subject-mean-fd-threshold", default=0.25, type=float)
    ap.add_argument("--subject-min-bad-runs", default=3, type=int)
    args = ap.parse_args()

    glm_dir = Path(args.glm_dir).expanduser().resolve()
    sens_path = Path(args.sensitivity_sets).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    tables = outdir / "tables"
    ensure_dir(outdir)
    ensure_dir(tables)

    sets = parse_list(args.sets)
    effects = parse_list(args.effects)

    run_contrasts = load_run_contrasts(glm_dir)
    run_inv = load_run_inventory(glm_dir)
    sens = load_sensitivity_sets(sens_path)

    qc_sub_path = Path(args.qc_subject_summary).expanduser().resolve() if args.qc_subject_summary else None
    qc_sub = load_qc_subject_summary(qc_sub_path) if qc_sub_path else None

    subject_motion = derive_subject_motion_summary(run_inv)
    high_motion_subjects, subject_motion = identify_high_motion_subjects(
        subject_motion,
        qc_subject_summary=qc_sub,
        mean_fd_threshold=args.subject_mean_fd_threshold,
        min_bad_runs=args.subject_min_bad_runs,
    )
    save_tsv(subject_motion, tables / "subject_motion_summary.tsv")

    all_subject_effects = []
    for set_name in sets:
        # motion_covariate uses main subject effects.
        source_set = "main" if set_name.startswith("motion_covariate") else set_name

        # Reuse the primary model when both primary and motion-covariate analyses are requested.
        existing_sets = [df["set_name"].iloc[0] for df in all_subject_effects if len(df) and "set_name" in df.columns]
        if source_set in existing_sets:
            continue

        allowed = build_allowed_runs(
            run_inv,
            sens,
            set_name=source_set,
            high_motion_subjects=high_motion_subjects,
            fd_bad_mean=args.fd_bad_mean,
            fd_bad_pct=args.fd_bad_pct,
        )
        se = compute_subject_condition_fe(run_contrasts, allowed, outdir=outdir, set_name=source_set)
        all_subject_effects.append(se)

    subject_effects = pd.concat(all_subject_effects, ignore_index=True) if all_subject_effects else pd.DataFrame()
    save_tsv(subject_effects, tables / "subject_effect_inventory.tsv")

    group_results, failures = run_group_level(
        subject_effects,
        effects=effects,
        set_names=sets,
        outdir=outdir,
        subject_motion=subject_motion,
        min_subjects=args.min_subjects,
    )
    save_tsv(group_results, tables / "group_level_results.tsv")
    save_tsv(failures, tables / "group_level_failures.tsv")

    robust = robustness_summary(group_results)
    save_tsv(robust, tables / "robustness_summary.tsv")

    manifest = {
        "timestamp_local": now_local_iso(),
        "script": Path(__file__).name,
        "python_version": sys.version.replace("\\n", " "),
        "platform": platform.platform(),
        "glm_dir": str(glm_dir),
        "glm_run_contrast_inventory": str(glm_dir / "tables" / "run_contrast_inventory.tsv"),
        "sensitivity_sets": str(sens_path),
        "sensitivity_sets_sha256": sha256_file(sens_path),
        "qc_subject_summary": str(qc_sub_path) if qc_sub_path else "",
        "outdir": str(outdir),
        "sets": sets,
        "effects": effects,
        "min_subjects": int(args.min_subjects),
        "fd_bad_mean": float(args.fd_bad_mean),
        "fd_bad_pct": float(args.fd_bad_pct),
        "subject_mean_fd_threshold": float(args.subject_mean_fd_threshold),
        "subject_min_bad_runs": int(args.subject_min_bad_runs),
        "high_motion_subjects": high_motion_subjects,
    }
    save_json(manifest, outdir / "manifest.json")
    write_report(outdir, manifest, subject_motion, subject_effects, group_results, robust)

    print("Processing completed successfully.")
    print(f"High-motion subjects: {len(high_motion_subjects)}")
    if high_motion_subjects:
        print(", ".join(high_motion_subjects))
    print(f"Subject-effect rows: {len(subject_effects)}")
    print(f"Group result rows: {len(group_results)}")
    if len(group_results):
        print(group_results[["set_name", "effect", "status", "n_subjects", "z_max_abs"]].to_string(index=False))
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
