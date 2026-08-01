#!/usr/bin/env python3
"""Verify regenerated compact-table workflows against archived results."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SCREENING_TABLES = (
    "pathway_behavior_correlations.tsv",
    "pathway_score_tests.tsv",
    "pathway_scores.tsv",
    "roi_behavior_correlations.tsv",
    "targeted_ec_edges.tsv",
    "targeted_ec_tests.tsv",
    "targeted_family_summary.tsv",
    "targeted_fc_edges.tsv",
    "targeted_fc_tests.tsv",
    "targeted_roi_activity_long.tsv",
    "targeted_roi_activity_tests.tsv",
)

AUDIT_GROUPS = {
    "passive-run-position": "passive_run_position_controls",
    "neural-matched-passive": "neural_matched_passive_roi",
    "nps-contrast-consistency": "nps_contrast_consistency",
    "nps-participant-influence": "nps_participant_influence",
    "roi-inference-qc": "roi_inference_qc",
}


def compare_tsv(generated: Path, archived: Path, *, atol: float = 1e-12, rtol: float = 1e-10) -> list[str]:
    errors: list[str] = []
    if not generated.is_file():
        return [f"Missing regenerated table: {generated}"]
    if not archived.is_file():
        return [f"Missing archived table: {archived}"]

    left = pd.read_csv(generated, sep="\t", low_memory=False)
    right = pd.read_csv(archived, sep="\t", low_memory=False)

    if list(left.columns) != list(right.columns):
        return [f"Column mismatch: {generated.name}"]
    if left.shape != right.shape:
        return [f"Shape mismatch: {generated.name}: {left.shape} != {right.shape}"]

    for column in left.columns:
        a = left[column]
        b = right[column]
        a_num = pd.to_numeric(a, errors="coerce")
        b_num = pd.to_numeric(b, errors="coerce")
        numeric_mask = a.notna() | b.notna()
        numeric = bool(numeric_mask.any()) and a_num[numeric_mask].notna().all() and b_num[numeric_mask].notna().all()
        if numeric:
            if not np.allclose(
                a_num.to_numpy(dtype=float),
                b_num.to_numpy(dtype=float),
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            ):
                delta = np.nanmax(np.abs(a_num.to_numpy(dtype=float) - b_num.to_numpy(dtype=float)))
                errors.append(f"Numeric mismatch: {generated.name}:{column}; max_abs_diff={delta:.6g}")
        else:
            a_text = a.fillna("<NA>").astype(str).tolist()
            b_text = b.fillna("<NA>").astype(str).tolist()
            if a_text != b_text:
                errors.append(f"Text or row-order mismatch: {generated.name}:{column}")
    return errors


def verify_screening(generated_root: Path) -> list[str]:
    errors: list[str] = []
    archived_root = ROOT / "results" / "trialwise_connectivity"
    for name in SCREENING_TABLES:
        errors.extend(compare_tsv(generated_root / "tables" / name, archived_root / name))
    return errors


def verify_audits(generated_root: Path) -> list[str]:
    errors: list[str] = []
    archived_base = ROOT / "results" / "sensitivity_audits"
    for generated_dir, archived_dir in AUDIT_GROUPS.items():
        generated_path = generated_root / generated_dir
        archived_path = archived_base / archived_dir
        if not generated_path.is_dir():
            errors.append(f"Missing regenerated audit directory: {generated_path}")
            continue
        for table in sorted(archived_path.glob("*.tsv")):
            errors.extend(compare_tsv(generated_path / table.name, table))
    return errors


def verify_smoke(generated_root: Path) -> list[str]:
    table = generated_root / "selected_score_reproduction.tsv"
    if not table.is_file():
        return [f"Missing smoke-test reproduction table: {table}"]
    frame = pd.read_csv(table, sep="\t")
    required = "forward_composite_vs_repository_selected_score"
    row = frame.loc[frame["check"].eq(required)]
    if len(row) != 1:
        return [f"Expected one '{required}' row in {table}"]
    value = float(row.iloc[0]["value"])
    if not np.isfinite(value) or value < 0.999999:
        return [f"Selected-score reproduction below tolerance: r={value}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("screening", "audits", "smoke"))
    parser.add_argument("--generated", type=Path, required=True)
    args = parser.parse_args()

    generated = args.generated.resolve()
    if args.mode == "screening":
        errors = verify_screening(generated)
    elif args.mode == "audits":
        errors = verify_audits(generated)
    else:
        errors = verify_smoke(generated)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Reproduced-output verification passed: {args.mode}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
