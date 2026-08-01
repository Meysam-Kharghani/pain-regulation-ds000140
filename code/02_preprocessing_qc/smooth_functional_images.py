#!/usr/bin/env python3
"""
smooth_functional_images.py

Post-fMRIPrep spatial smoothing for ds000140.

Implementation notes
--------------------
- If the QC table points to a native-space brain mask while the BOLD is in MNI space,
  automatically locate the space-matched fMRIPrep brain mask from the BOLD filename.
- Record the originally supplied mask and the actual mask used in the inventory.

Design principles
-----------------
Scope:
  - Create smoothed BOLD images from fMRIPrep preprocessed BOLD files.

Out of scope:
  - Re-run QC decisions.
  - Exclude subjects/runs using a new policy.
  - Create PSC images.
  - Create GLM confound files.
  - Create multiple smoothing levels by default.

The script reads the QC inclusion table produced by:
  run_preprocessing_qc.py

Runs are processed when:
  keep_main == True

Default output:
  derivatives/smoothed_fwhm6/

Main use
--------
Use the smoothed FWHM=6 mm BOLD files for main univariate first-level GLM.
Retain the original unsmoothed fMRIPrep BOLD files for pattern- and ROI-sensitive
analyses, including NPS, beta-series connectivity, and small-region analyses.

Example
-------
python smooth_functional_images.py \
  --fmriprep-dir <PROJECT_ROOT>/derivatives/fmriprep \
  --qc-inclusion <PROJECT_ROOT>/derivatives/qc_fmriprep_analysis/tables/analysis_inclusion_table.tsv \
  --outdir <PROJECT_ROOT>/derivatives/smoothed_fwhm6 \
  --fwhm 6

From inside derivatives/fmriprep:
python smooth_functional_images.py \
  --fmriprep-dir . \
  --qc-inclusion ../qc_fmriprep_analysis/tables/analysis_inclusion_table.tsv \
  --outdir ../smoothed_fwhm6 \
  --fwhm 6
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
from scipy.ndimage import gaussian_filter


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


def parse_fwhm_list(x: str) -> List[float]:
    vals = []
    for part in str(x).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError("At least one FWHM value is required.")
    return vals


def fwhm_tag(fwhm: float) -> str:
    """Return a clean filename tag such as smoothedFWHM6 or smoothedFWHM6p5."""
    if abs(fwhm - round(fwhm)) < 1e-8:
        return f"smoothedFWHM{int(round(fwhm))}"
    s = f"{fwhm:g}".replace(".", "p")
    return f"smoothedFWHM{s}"


def bool_from_any(x) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def sidecar_json_for_niigz(path: Path) -> Path:
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name.replace(".nii.gz", ".json"))
    if name.endswith(".nii"):
        return path.with_name(name.replace(".nii", ".json"))
    return path.with_suffix(".json")


def expected_space_mask_from_bold(bold_path: Path) -> Path:
    """Return the fMRIPrep brain mask expected to match a space-specific preproc BOLD.

    Example:
      sub-01_task-X_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
      ->
      sub-01_task-X_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
    """
    name = bold_path.name
    if name.endswith("_desc-preproc_bold.nii.gz"):
        return bold_path.with_name(name.replace("_desc-preproc_bold.nii.gz", "_desc-brain_mask.nii.gz"))
    if "desc-preproc_bold" in name:
        return bold_path.with_name(name.replace("desc-preproc_bold", "desc-brain_mask"))
    return bold_path


def resolve_mask_for_bold(bold_path: Path, mask_path_from_qc: Path) -> tuple[Path, str]:
    """Choose a brain mask with the same grid as the BOLD.

    The QC table may contain the native-space run mask, e.g.
      sub-XX_task-..._run-YY_desc-brain_mask.nii.gz
    while the BOLD is in MNI space, e.g.
      sub-XX_task-..._run-YY_space-MNI..._desc-preproc_bold.nii.gz

    This function first tries the BOLD-matched MNI-space mask, then the QC-table
    mask. Shape checking is still done later before smoothing.
    """
    expected = expected_space_mask_from_bold(bold_path)
    if expected.exists():
        if expected.resolve() == mask_path_from_qc.resolve():
            return expected, "qc_mask_already_space_matched"
        return expected, "auto_space_matched_mask_from_bold"

    if mask_path_from_qc.exists():
        return mask_path_from_qc, "qc_table_mask_used_no_space_matched_mask_found"

    return mask_path_from_qc, "qc_table_mask_missing"


def get_rel_or_abs(path_str: str, base: Optional[Path] = None) -> Path:
    p = Path(str(path_str)).expanduser()
    if p.is_absolute():
        return p.resolve()
    if base is not None:
        return (base / p).resolve()
    return p.resolve()


# =============================================================================
# QC inclusion table handling
# =============================================================================

REQUIRED_QC_COLUMNS = [
    "subject", "run", "keep_main", "analysis_decision",
    "preproc_bold", "brain_mask", "confounds_tsv",
]


def load_qc_inclusion(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    missing = [c for c in REQUIRED_QC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"QC inclusion table is missing required columns: {missing}\n"
            f"Expected file from run_preprocessing_qc.py: tables/analysis_inclusion_table.tsv"
        )

    df = df.copy()
    df["subject"] = df["subject"].astype(str)
    df["run"] = pd.to_numeric(df["run"], errors="coerce").astype("Int64")
    df["keep_main_bool"] = df["keep_main"].map(bool_from_any)
    if "sensitivity_exclude_candidate" in df.columns:
        df["sensitivity_exclude_candidate_bool"] = df["sensitivity_exclude_candidate"].map(bool_from_any)
    else:
        df["sensitivity_exclude_candidate_bool"] = False

    return df


def filter_runs(df: pd.DataFrame, include_mode: str) -> pd.DataFrame:
    """Filter based on QC decision."""
    include_mode = include_mode.lower().strip()
    if include_mode == "keep_main":
        return df[df["keep_main_bool"]].copy()
    if include_mode == "all_qc_rows":
        return df.copy()
    if include_mode == "strict_no_sensitivity_candidates":
        return df[df["keep_main_bool"] & (~df["sensitivity_exclude_candidate_bool"])].copy()
    raise ValueError(
        "Unknown include mode. Use one of: "
        "keep_main, all_qc_rows, strict_no_sensitivity_candidates"
    )


# =============================================================================
# Output naming
# =============================================================================

def infer_output_func_dir(outdir: Path, bold_path: Path, subject: str) -> Path:
    """Mirror an fMRIPrep-style participant/functional structure."""
    return outdir / subject / "func"


def make_smoothed_output_path(outdir: Path, bold_path: Path, subject: str, fwhm: float) -> Path:
    func_dir = infer_output_func_dir(outdir, bold_path, subject)
    tag = fwhm_tag(fwhm)
    name = bold_path.name

    # Standard fMRIPrep pattern:
    # sub-XX_task-..._run-YY_space-..._desc-preproc_bold.nii.gz
    if name.endswith("_desc-preproc_bold.nii.gz"):
        out_name = name.replace("_desc-preproc_bold.nii.gz", f"_desc-{tag}_bold.nii.gz")
    elif name.endswith("_bold.nii.gz"):
        out_name = name.replace("_bold.nii.gz", f"_desc-{tag}_bold.nii.gz")
    else:
        out_name = re.sub(r"\.nii(\.gz)?$", f"_desc-{tag}_bold.nii.gz", name)

    return func_dir / out_name


# =============================================================================
# Smoothing
# =============================================================================

def voxel_sizes_from_affine(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def fwhm_to_sigma_vox(fwhm_mm: float, affine: np.ndarray) -> Tuple[float, float, float]:
    sigma_mm = fwhm_mm / math.sqrt(8.0 * math.log(2.0))
    vox = voxel_sizes_from_affine(affine)
    return tuple((sigma_mm / vox).tolist())


def smooth_4d_masked(
    bold_path: Path,
    mask_path: Path,
    out_path: Path,
    fwhm_mm: float,
    dtype: np.dtype = np.float32,
) -> Dict[str, object]:
    """Smooth 4D image volume-by-volume and zero out voxels outside mask.

    This avoids creating PSC images and avoids re-estimating any QC.
    It uses scipy.ndimage.gaussian_filter in voxel space, with sigma derived
    from image affine and requested FWHM in millimeters.
    """
    img = nib.load(str(bold_path))
    mask_img = nib.load(str(mask_path))

    if len(img.shape) != 4:
        raise ValueError(f"Expected 4D BOLD image; got shape {img.shape}: {bold_path}")

    if mask_img.shape[:3] != img.shape[:3]:
        raise ValueError(
            f"Mask/BOLD shape mismatch:\n"
            f"  BOLD: {bold_path} shape={img.shape[:3]}\n"
            f"  MASK: {mask_path} shape={mask_img.shape[:3]}"
        )

    sigma_vox = fwhm_to_sigma_vox(fwhm_mm, img.affine)
    mask = np.asanyarray(mask_img.dataobj).astype(bool)

    n_x, n_y, n_z, n_t = img.shape
    out = np.zeros((n_x, n_y, n_z, n_t), dtype=dtype)

    # Access data lazily via proxy; process one 3D volume at a time.
    proxy = img.dataobj
    for t in range(n_t):
        vol = np.asanyarray(proxy[..., t]).astype(np.float32)
        sm = gaussian_filter(vol, sigma=sigma_vox, mode="nearest")
        sm[~mask] = 0.0
        out[..., t] = sm.astype(dtype, copy=False)

    header = img.header.copy()
    header.set_data_dtype(dtype)
    out_img = nib.Nifti1Image(out, img.affine, header)
    ensure_dir(out_path.parent)
    nib.save(out_img, str(out_path))

    return {
        "n_vols": int(n_t),
        "shape": list(img.shape),
        "voxel_size_mm": [float(v) for v in voxel_sizes_from_affine(img.affine)],
        "fwhm_mm": float(fwhm_mm),
        "sigma_vox": [float(x) for x in sigma_vox],
        "mask_voxels": int(mask.sum()),
    }


def write_smoothed_json(
    out_img_path: Path,
    bold_path: Path,
    mask_path: Path,
    fwhm: float,
    smoothing_info: Dict[str, object],
    script_name: str,
) -> None:
    sidecar = sidecar_json_for_niigz(out_img_path)
    meta = {
        "GeneratedBy": script_name,
        "GeneratedOn": now_local_iso(),
        "Description": "Spatially smoothed fMRIPrep preprocessed BOLD image.",
        "Sources": [str(bold_path)],
        "BrainMask": str(mask_path),
        "SpatialSmoothingFWHMmm": float(fwhm),
        "SmoothingKernel": "Gaussian",
        "SmoothingImplementation": "scipy.ndimage.gaussian_filter, volume-wise, sigma derived from affine voxel sizes",
        "MaskingAfterSmoothing": "Voxels outside the fMRIPrep brain mask were set to zero after smoothing.",
        "IntensityScaling": "none",
        "RecommendedUse": "Main univariate first-level GLM / group-level GLM.",
        "NotRecommendedUse": "Pattern-expression, small-ROI, NPS, beta-series connectivity; use unsmoothed fMRIPrep BOLD for those analyses.",
        "SmoothingInfo": smoothing_info,
    }
    save_json(meta, sidecar)


# =============================================================================
# Run processing
# =============================================================================

def process_one_row(
    row: pd.Series,
    outdir: Path,
    fwhm_values: Sequence[float],
    overwrite: bool,
    dry_run: bool,
    script_name: str,
) -> List[Dict[str, object]]:
    subject = str(row["subject"])
    run = int(row["run"])
    bold_path = get_rel_or_abs(row["preproc_bold"])
    mask_path_from_qc = get_rel_or_abs(row["brain_mask"])
    mask_path, mask_source = resolve_mask_for_bold(bold_path, mask_path_from_qc)
    confounds_path = get_rel_or_abs(row["confounds_tsv"])

    outputs = []

    # Basic file checks
    missing = []
    if not bold_path.exists():
        missing.append(f"preproc_bold_missing:{bold_path}")
    if not mask_path.exists():
        missing.append(f"brain_mask_missing:{mask_path}")
    if not confounds_path.exists():
        missing.append(f"confounds_missing:{confounds_path}")

    if missing:
        return [{
            "subject": subject,
            "run": run,
            "fwhm": np.nan,
            "status": "FAILED",
            "error": "; ".join(missing),
            "bold_unsmoothed": str(bold_path),
            "bold_smoothed": "",
            "brain_mask": str(mask_path),
            "brain_mask_from_qc": str(mask_path_from_qc),
            "brain_mask_source": mask_source,
            "confounds_tsv": str(confounds_path),
            "analysis_decision": row.get("analysis_decision", ""),
            "keep_main": row.get("keep_main", ""),
            "sensitivity_exclude_candidate": row.get("sensitivity_exclude_candidate", ""),
        }]

    for fwhm in fwhm_values:
        out_img = make_smoothed_output_path(outdir, bold_path, subject, fwhm)
        status = "SKIPPED_EXISTS"
        error = ""
        info = {}

        try:
            if dry_run:
                status = "DRY_RUN"
            elif out_img.exists() and not overwrite:
                status = "SKIPPED_EXISTS"
                # Read the existing image shape when available.
                try:
                    img = nib.load(str(out_img))
                    info = {"n_vols": int(img.shape[3]) if len(img.shape) == 4 else np.nan, "shape": list(img.shape)}
                except Exception:
                    info = {}
            else:
                info = smooth_4d_masked(
                    bold_path=bold_path,
                    mask_path=mask_path,
                    out_path=out_img,
                    fwhm_mm=float(fwhm),
                )
                write_smoothed_json(
                    out_img_path=out_img,
                    bold_path=bold_path,
                    mask_path=mask_path,
                    fwhm=float(fwhm),
                    smoothing_info=info,
                    script_name=script_name,
                )
                status = "OK"
        except Exception as e:
            status = "FAILED"
            error = repr(e)

        outputs.append({
            "subject": subject,
            "run": run,
            "fwhm": float(fwhm),
            "status": status,
            "error": error,
            "bold_unsmoothed": str(bold_path),
            "bold_smoothed": str(out_img) if status != "FAILED" else "",
            "brain_mask": str(mask_path),
            "brain_mask_from_qc": str(mask_path_from_qc),
            "brain_mask_source": mask_source,
            "confounds_tsv": str(confounds_path),
            "analysis_decision": row.get("analysis_decision", ""),
            "keep_main": row.get("keep_main", ""),
            "sensitivity_exclude_candidate": row.get("sensitivity_exclude_candidate", ""),
            "motion_flag": row.get("motion_flag", ""),
            "nuisance_burden_flag": row.get("nuisance_burden_flag", ""),
            "file_flag": row.get("file_flag", ""),
            "fd_mean": row.get("fd_mean", np.nan),
            "fd_max": row.get("fd_max", np.nan),
            "pct_fd_gt_0p5": row.get("pct_fd_gt_0p5", np.nan),
            "motion_outlier_volumes_pct": row.get("motion_outlier_volumes_pct", np.nan),
            "estimated_dof_after_nuisance": row.get("estimated_dof_after_nuisance", np.nan),
            "n_vols": info.get("n_vols", np.nan) if isinstance(info, dict) else np.nan,
            "shape": json.dumps(info.get("shape", [])) if isinstance(info, dict) else "[]",
        })

    return outputs


# =============================================================================
# Report
# =============================================================================

def write_report(
    outdir: Path,
    manifest: Dict[str, object],
    inventory: pd.DataFrame,
    qc_used: pd.DataFrame,
) -> None:
    n_runs_unique = qc_used[["subject", "run"]].drop_duplicates().shape[0]
    n_subs = qc_used["subject"].nunique()
    status_counts = inventory["status"].value_counts().to_dict() if len(inventory) else {}

    lines = []
    lines.append("# Spatial smoothing report\n\n")
    lines.append("## Overview\n")
    lines.append(f"- Generated: {manifest['timestamp_local']}\n")
    lines.append(f"- fMRIPrep directory: `{manifest['fmriprep_dir']}`\n")
    lines.append(f"- QC inclusion table: `{manifest['qc_inclusion']}`\n")
    lines.append(f"- Output directory: `{manifest['outdir']}`\n")
    lines.append(f"- FWHM values: {manifest['fwhm_values']}\n")
    lines.append(f"- Include mode: `{manifest['include_mode']}`\n")
    lines.append(f"- Subjects processed: {n_subs}\n")
    lines.append(f"- Runs selected: {n_runs_unique}\n\n")

    lines.append("## Status counts\n")
    for k, v in sorted(status_counts.items()):
        lines.append(f"- {k}: {v}\n")
    lines.append("\n")

    lines.append("## Design policy\n")
    lines.append("- This script only performs spatial smoothing.\n")
    lines.append("- QC decisions are imported from the fMRIPrep QC table; no new exclusion rule is applied here.\n")
    lines.append("- No PSC conversion is performed.\n")
    lines.append("- No GLM confound file is created here; confound selection and design-rank/DOF checks are performed in the first-level GLM preparation step.\n")
    lines.append("- Default FWHM=6 mm is intended for the main univariate GLM.\n")
    lines.append("- Use unsmoothed fMRIPrep BOLD for pattern-expression, small ROI, NPS, PAG/NAc, or beta-series connectivity analyses.\n\n")

    lines.append("## Key outputs\n")
    lines.append("- `tables/smoothing_inventory.tsv`\n")
    lines.append("- `tables/smoothing_failed.tsv`\n")
    lines.append("- `tables/qc_rows_used_for_smoothing.tsv`\n")
    lines.append("- Smoothed images in subject/func subfolders.\n")

    (outdir / "smoothing_report.md").write_text("".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean fMRIPrep BOLD smoothing based on  QC inclusion table.")
    parser.add_argument("--fmriprep-dir", required=True, type=str, help="Path to derivatives/fmriprep. Used for manifest/provenance.")
    parser.add_argument("--qc-inclusion", required=True, type=str, help="Path to qc_fmriprep_analysis/tables/analysis_inclusion_table.tsv")
    parser.add_argument("--outdir", required=True, type=str, help="Output derivative directory, e.g. derivatives/smoothed_fwhm6")
    parser.add_argument("--fwhm", default="6", type=str, help="Comma-separated FWHM values in mm. Default: 6. Example: 6 or 6,3")
    parser.add_argument(
        "--include-mode",
        default="keep_main",
        choices=["keep_main", "strict_no_sensitivity_candidates", "all_qc_rows"],
        help=(
            "Which QC rows to smooth. Default keep_main. "
            "strict_no_sensitivity_candidates smooths only clean/non-sensitivity rows. "
            "all_qc_rows smooths everything in the QC table and is not recommended for main analysis."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing smoothed images.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write images; only create planned inventory.")
    args = parser.parse_args()

    fmriprep_dir = Path(args.fmriprep_dir).expanduser().resolve()
    qc_path = Path(args.qc_inclusion).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    tables = outdir / "tables"
    ensure_dir(outdir)
    ensure_dir(tables)

    fwhm_values = parse_fwhm_list(args.fwhm)

    qc = load_qc_inclusion(qc_path)
    qc_used = filter_runs(qc, args.include_mode).sort_values(["subject", "run"]).reset_index(drop=True)

    save_tsv(qc_used, tables / "qc_rows_used_for_smoothing.tsv")

    inventory_rows: List[Dict[str, object]] = []
    script_name = Path(__file__).name

    for _, row in qc_used.iterrows():
        inventory_rows.extend(
            process_one_row(
                row=row,
                outdir=outdir,
                fwhm_values=fwhm_values,
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
                script_name=script_name,
            )
        )

    inv = pd.DataFrame(inventory_rows)
    save_tsv(inv, tables / "smoothing_inventory.tsv")

    failed = inv[inv["status"] == "FAILED"].copy() if len(inv) else pd.DataFrame()
    save_tsv(failed, tables / "smoothing_failed.tsv")

    manifest = {
        "timestamp_local": now_local_iso(),
        "script": script_name,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "fmriprep_dir": str(fmriprep_dir),
        "qc_inclusion": str(qc_path),
        "qc_inclusion_sha256": sha256_file(qc_path),
        "outdir": str(outdir),
        "fwhm_values": fwhm_values,
        "include_mode": args.include_mode,
        "overwrite": bool(args.overwrite),
        "dry_run": bool(args.dry_run),
        "n_qc_rows_total": int(qc.shape[0]),
        "n_qc_rows_selected": int(qc_used.shape[0]),
        "n_subjects_selected": int(qc_used["subject"].nunique()),
        "n_outputs_planned_or_written": int(inv.shape[0]),
        "n_failed": int((inv["status"] == "FAILED").sum()) if len(inv) else 0,
        "notes": [
            "This script performs spatial smoothing only.",
            "QC decisions are imported from analysis_inclusion_table.tsv.",
            "No PSC conversion is performed.",
            "No confound selection is performed here.",
            "Use FWHM=6 mm smoothed BOLD for main univariate GLM.",
            "Use unsmoothed fMRIPrep BOLD for pattern-expression, small-ROI, and connectivity analyses.",
        ],
    }
    save_json(manifest, outdir / "manifest.json")

    write_report(outdir, manifest, inv, qc_used)

    print("Processing completed successfully.")
    print(f"QC rows total: {qc.shape[0]}")
    print(f"Rows selected for smoothing: {qc_used.shape[0]}")
    print(f"Output rows: {inv.shape[0]}")
    print(f"Failed: {manifest['n_failed']}")
    print(f"Output directory: {outdir}")
    print(f"Inventory: {tables / 'smoothing_inventory.tsv'}")


if __name__ == "__main__":
    main()
