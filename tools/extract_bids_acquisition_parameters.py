#!/usr/bin/env python3
"""Extract acquisition parameters from BIDS JSON sidecars and NIfTI headers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import pandas as pd


def inherited_json_candidates(path: Path, dataset_root: Path) -> list[Path]:
    """Return BIDS sidecars from least to most specific."""
    name = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    adjacent = path.with_name(name + ".json")
    task = None
    for token in name.split("_"):
        if token.startswith("task-"):
            task = token
            break
    candidates: list[Path] = []
    if task:
        candidates.extend([
            dataset_root / f"{task}_bold.json",
            path.parent.parent / f"{task}_bold.json",
            path.parent / f"{task}_bold.json",
        ])
    candidates.append(adjacent)
    seen: set[Path] = set()
    return [c for c in candidates if not (c in seen or seen.add(c)) and c.exists()]


def read_inherited_metadata(path: Path, dataset_root: Path) -> tuple[dict, list[Path]]:
    metadata: dict = {}
    sidecars = inherited_json_candidates(path, dataset_root)
    for sidecar in sidecars:
        metadata.update(json.loads(sidecar.read_text(encoding="utf-8")))
    return metadata, sidecars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.dataset_root.glob("sub-*/func/*_bold.nii*"))
    if not files:
        raise FileNotFoundError("No BIDS BOLD NIfTI files found.")

    rows = []
    for nifti in files:
        img = nib.load(str(nifti))
        zooms = img.header.get_zooms()
        metadata, sidecars = read_inherited_metadata(nifti, args.dataset_root)
        rows.append({
            "nifti": nifti.relative_to(args.dataset_root).as_posix(),
            "json": " | ".join(p.relative_to(args.dataset_root).as_posix() for p in sidecars),
            "matrix_x": int(img.shape[0]),
            "matrix_y": int(img.shape[1]),
            "n_slices": int(img.shape[2]),
            "n_volumes": int(img.shape[3]) if len(img.shape) > 3 else 1,
            "voxel_x_mm": float(zooms[0]),
            "voxel_y_mm": float(zooms[1]),
            "voxel_z_mm": float(zooms[2]),
            "header_tr_sec": float(zooms[3]) if len(zooms) > 3 else None,
            "RepetitionTime": metadata.get("RepetitionTime"),
            "EchoTime": metadata.get("EchoTime"),
            "MagneticFieldStrength": metadata.get("MagneticFieldStrength"),
            "Manufacturer": metadata.get("Manufacturer"),
            "ManufacturersModelName": metadata.get("ManufacturersModelName"),
            "SequenceName": metadata.get("SequenceName"),
            "PulseSequenceType": metadata.get("PulseSequenceType"),
            "PhaseEncodingDirection": metadata.get("PhaseEncodingDirection"),
            "FlipAngle": metadata.get("FlipAngle"),
            "EffectiveEchoSpacing": metadata.get("EffectiveEchoSpacing"),
            "MultibandAccelerationFactor": metadata.get("MultibandAccelerationFactor"),
            "ParallelReductionFactorInPlane": metadata.get("ParallelReductionFactorInPlane"),
            "n_slice_timing_values": len(metadata.get("SliceTiming", [])),
        })

    table = pd.DataFrame(rows)
    table.to_csv(args.outdir / "bold_acquisition_file_level.tsv", sep="\t", index=False)

    fields = [c for c in table.columns if c not in {"nifti", "json", "n_volumes"}]
    summary_rows = []
    for field in fields:
        values = table[field].dropna().astype(str).unique().tolist()
        summary_rows.append({
            "field": field,
            "n_unique": len(values),
            "unique_values": " | ".join(sorted(values)),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.outdir / "bold_acquisition_unique_values.tsv", sep="\t", index=False)

    def unique(field: str) -> str:
        values = table[field].dropna().unique().tolist()
        return ", ".join(str(v) for v in values) if values else "not available"

    sentence = (
        "Functional MRI acquisition parameters recovered from the released BIDS files: "
        f"field strength {unique('MagneticFieldStrength')} T; "
        f"TR {unique('RepetitionTime')} s; TE {unique('EchoTime')} s; "
        f"matrix {unique('matrix_x')} × {unique('matrix_y')}; "
        f"{unique('n_slices')} slices; voxel size "
        f"{unique('voxel_x_mm')} × {unique('voxel_y_mm')} × {unique('voxel_z_mm')} mm."
    )
    (args.outdir / "acquisition_summary_sentence.txt").write_text(sentence + "\n", encoding="utf-8")
    print(sentence)


if __name__ == "__main__":
    main()
