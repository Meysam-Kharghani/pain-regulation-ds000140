#!/usr/bin/env python3
"""
Prepare ROI masks and optional restricted pain-signature resources for analysis.

Expected NPS zip contents:
  weights_NSF_grouppred_cvpcr.hdr
  weights_NSF_grouppred_cvpcr.img.gz

Important:
  Restricted third-party signature weight maps are not redistributed in this repository.
  Users must supply such resources locally when permitted by the applicable license or data-use agreement.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_existing_roi_dir(roi_dir: Path, outdir: Path, overwrite: bool = False) -> None:
    if outdir.exists() and overwrite:
        shutil.rmtree(outdir)

    ensure_dir(outdir)

    for fname in [
        "roi_definitions.tsv",
        "roi_mask_build_report.tsv",
        "roi_label_matches.tsv",
        "signature_definitions.tsv",
        "manifest.json",
        "roi_masks_report.md",
    ]:
        src = roi_dir / fname
        if src.exists():
            shutil.copy2(src, outdir / fname)

    for dname in ["masks", "signatures", "label_diagnostics"]:
        src = roi_dir / dname
        dst = outdir / dname
        if src.exists():
            if dst.exists() and overwrite:
                shutil.rmtree(dst)
            if not dst.exists():
                shutil.copytree(src, dst)


def gunzip_img_if_needed(img_gz: Path) -> Path:
    if not img_gz.name.endswith(".img.gz"):
        return img_gz

    img_path = img_gz.with_suffix("")  # .img.gz -> .img
    if img_path.exists() and img_path.stat().st_size > 0:
        return img_path

    with gzip.open(img_gz, "rb") as f_in, open(img_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return img_path


def locate_required_nps_files(restricted_source_dir: Path) -> tuple[Path, Path]:
    hdr_hits = list(restricted_source_dir.rglob("weights_NSF_grouppred_cvpcr.hdr"))
    if not hdr_hits:
        raise FileNotFoundError("Could not find weights_NSF_grouppred_cvpcr.hdr in the NPS zip.")

    hdr = hdr_hits[0]
    folder = hdr.parent

    img_gz = folder / "weights_NSF_grouppred_cvpcr.img.gz"
    img = folder / "weights_NSF_grouppred_cvpcr.img"

    if img_gz.exists():
        img = gunzip_img_if_needed(img_gz)
    elif not img.exists():
        raise FileNotFoundError("Could not find weights_NSF_grouppred_cvpcr.img.gz or .img in the NPS zip.")

    return hdr, img


def resample_weight_map_to_reference(source_img: nib.spatialimages.SpatialImage,
                                     ref_img: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    res = resample_to_img(
        source_img,
        ref_img,
        interpolation="continuous",
        force_resample=True,
        copy_header=True,
    )
    data = np.asanyarray(res.dataobj).astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return nib.Nifti1Image(data, ref_img.affine, ref_img.header.copy())


def add_or_update_signature(sig_df: pd.DataFrame, row: dict) -> pd.DataFrame:
    sig_df = sig_df.copy()

    for k in row:
        if k not in sig_df.columns:
            sig_df[k] = np.nan

    if "signature" in sig_df.columns and row["signature"] in set(sig_df["signature"].astype(str)):
        idx = sig_df.index[sig_df["signature"].astype(str) == row["signature"]]
        for k, v in row.items():
            sig_df.loc[idx, k] = v
    else:
        sig_df = pd.concat([sig_df, pd.DataFrame([row])], ignore_index=True)

    return sig_df


def process_one_signature(source_hdr: Path,
                          ref_img: nib.spatialimages.SpatialImage,
                          out_map: Path,
                          signature_name: str,
                          family: str,
                          note: str) -> dict:
    source_img = nib.load(str(source_hdr))
    resampled = resample_weight_map_to_reference(source_img, ref_img)

    ensure_dir(out_map.parent)
    resampled.to_filename(str(out_map))

    data = np.asanyarray(resampled.dataobj)
    n_nonzero = int(np.sum(np.abs(data) > 1e-12))

    return {
        "signature": signature_name,
        "family": family,
        "status": "OK" if n_nonzero > 0 else "EMPTY_AFTER_RESAMPLE",
        "source_file": str(source_hdr),
        "resampled_map": str(out_map),
        "n_nonzero_voxels": n_nonzero,
        "note": note,
    }


def maybe_process_component(restricted_source_dir: Path,
                            base_name: str,
                            ref_img: nib.spatialimages.SpatialImage,
                            out_map: Path,
                            signature_name: str,
                            note: str) -> dict:
    hdr_hits = list(restricted_source_dir.rglob(base_name + ".hdr"))
    if not hdr_hits:
        return {
            "signature": signature_name,
            "family": "signature_component",
            "status": "MISSING",
            "source_file": "",
            "resampled_map": "",
            "n_nonzero_voxels": np.nan,
            "note": note,
        }

    hdr = hdr_hits[0]
    folder = hdr.parent
    img_gz = folder / (base_name + ".img.gz")
    if img_gz.exists():
        gunzip_img_if_needed(img_gz)

    return process_one_signature(
        source_hdr=hdr,
        ref_img=ref_img,
        out_map=out_map,
        signature_name=signature_name,
        family="signature_component",
        note=note,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nps-zip", required=True, help="Path to NPS_share.zip")
    parser.add_argument("--roi-dir", required=True, help="Existing roi_masks_analysis directory")
    parser.add_argument("--reference-img", required=True, help="Reference GLM image in the target space/grid")
    parser.add_argument("--outdir", required=True, help="Output directory, e.g., roi_masks_analysis_NPS")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outdir if it exists")
    parser.add_argument("--include-components", action="store_true",
                        help="Also include thresholded positive/negative NPS component maps.")
    args = parser.parse_args()

    nps_zip = Path(args.nps_zip).expanduser().resolve()
    roi_dir = Path(args.roi_dir).expanduser().resolve()
    ref_path = Path(args.reference_img).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    if not nps_zip.exists():
        raise FileNotFoundError(f"NPS zip not found: {nps_zip}")
    if not roi_dir.exists():
        raise FileNotFoundError(f"ROI directory not found: {roi_dir}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")

    copy_existing_roi_dir(roi_dir, outdir, overwrite=args.overwrite)

    restricted_source_dir = outdir / "external_signature_source_restricted"
    ensure_dir(restricted_source_dir)

    with zipfile.ZipFile(nps_zip, "r") as z:
        z.extractall(restricted_source_dir)

    nps_hdr, nps_img = locate_required_nps_files(restricted_source_dir)

    ref_img = nib.load(str(ref_path))
    sig_dir = outdir / "signatures"
    ensure_dir(sig_dir)

    rows = []
    rows.append(
        process_one_signature(
            source_hdr=nps_hdr,
            ref_img=ref_img,
            out_map=sig_dir / "nps.nii.gz",
            signature_name="NPS",
            family="signature",
            note="Neurologic Pain Signature full weight map from Wager et al. 2013; provided under a restricted-use agreement; not redistributed here.",
        )
    )

    if args.include_components:
        rows.append(
            maybe_process_component(
                restricted_source_dir=restricted_source_dir,
                base_name="weights_NSF_positive_smoothed_larger_than_10vox",
                ref_img=ref_img,
                out_map=sig_dir / "nps_positive_component_thr10vox.nii.gz",
                signature_name="NPS_positive_component_thr10vox",
                note="Thresholded positive NPS component; component diagnostic only, not the full NPS.",
            )
        )
        rows.append(
            maybe_process_component(
                restricted_source_dir=restricted_source_dir,
                base_name="weights_NSF_negative_smoothed_larger_than_10vox",
                ref_img=ref_img,
                out_map=sig_dir / "nps_negative_component_thr10vox.nii.gz",
                signature_name="NPS_negative_component_thr10vox",
                note="Thresholded negative NPS component; component diagnostic only, not the full NPS.",
            )
        )

    sig_path = outdir / "signature_definitions.tsv"
    if sig_path.exists():
        sig_df = pd.read_csv(sig_path, sep="\t")
    else:
        sig_df = pd.DataFrame()

    for row in rows:
        sig_df = add_or_update_signature(sig_df, row)

    sig_df.to_csv(sig_path, sep="\t", index=False)

    manifest = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "nps_zip": str(nps_zip),
        "roi_dir_source": str(roi_dir),
        "reference_img": str(ref_path),
        "outdir": str(outdir),
        "nps_hdr": str(nps_hdr),
        "nps_img": str(nps_img),
        "nps_rows_added_or_updated": rows,
        "restrictions": [
            "NPS files are restricted third-party resources.",
            "Do not redistribute original or resampled signature maps unless permitted by the provider.",
            "Use restricted resources only according to the applicable data-use agreement.",
        ],
    }
    save_json(manifest, outdir / "manifest_nps_integration.json")

    lines = []
    lines.append("# NPS integration report\n\n")
    lines.append(f"- Generated: {manifest['timestamp_local']}\n")
    lines.append(f"- Source ROI directory: `{roi_dir}`\n")
    lines.append(f"- Output directory: `{outdir}`\n")
    lines.append(f"- Reference image: `{ref_path}`\n\n")
    lines.append("## Added / updated signatures\n")
    for row in rows:
        lines.append(
            f"- {row['signature']}: status={row['status']}, "
            f"nonzero_voxels={row['n_nonzero_voxels']}, "
            f"map=`{row['resampled_map']}`\n"
        )
    lines.append("\n## Restrictions\n")
    for r in manifest["restrictions"]:
        lines.append(f"- {r}\n")

    (outdir / "nps_integration_report.md").write_text("".join(lines), encoding="utf-8")

    print("Processing completed successfully.")
    print(f"Output directory: {outdir}")
    print(pd.DataFrame(rows)[["signature", "status", "n_nonzero_voxels", "resampled_map"]].to_string(index=False))


if __name__ == "__main__":
    main()
