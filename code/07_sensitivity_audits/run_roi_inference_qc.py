#!/usr/bin/env python3
"""Regenerate ROI multiplicity and run-inclusion quality-control tables.

The workflow applies Benjamini-Hochberg correction across all ROI tests within
an analysis set and contrast, summarizes available runs and participants, and
writes the evidence hierarchy used to delimit targeted, sensitivity, and
exploratory claims.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def read_table(path: Path) -> pd.DataFrame:
    if path.name.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def canonical_subject(value: object) -> str:
    text = str(value)
    match = re.search(r"sub[-_]?(\d+)", text, flags=re.I)
    if match:
        return f"sub-{int(match.group(1)):02d}"
    match = re.search(r"(\d+)", text)
    return f"sub-{int(match.group(1)):02d}" if match else text


def display_path(path: Path, project: Path) -> str:
    try:
        return path.resolve().relative_to(project.resolve()).as_posix()
    except ValueError:
        return path.name


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(values, np.nan)
    valid = np.isfinite(values)
    finite = values[valid]
    if finite.size == 0:
        return adjusted

    order = np.argsort(finite)
    ranked = finite[order]
    corrected = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.minimum(corrected, 1.0)

    reordered = np.empty_like(corrected)
    reordered[order] = corrected
    adjusted[valid] = reordered
    return adjusted


def standardize_roi_statistics(path: Path) -> pd.DataFrame:
    frame = read_table(path).copy()
    rename: dict[str, str] = {}

    for column in frame.columns:
        normalized = normalize_name(column)
        if normalized in {"analysis_set", "set_name", "set"}:
            rename[column] = "analysis_set"
        elif normalized in {"effect", "contrast"}:
            rename[column] = "effect"
        elif normalized in {"family", "roi_family"}:
            rename[column] = "family"
        elif normalized in {"name", "roi", "mask", "region"}:
            rename[column] = "roi"
        elif normalized == "mean":
            rename[column] = "mean"
        elif normalized in {"stat", "t", "z", "statistic"}:
            rename[column] = "stat"
        elif normalized in {"p", "p_value", "pval"}:
            rename[column] = "p"
        elif normalized in {"q", "q_fdr", "q_family", "fdr"}:
            rename[column] = "q_family"

    frame = frame.rename(columns=rename)
    required = [
        "analysis_set",
        "effect",
        "family",
        "roi",
        "mean",
        "stat",
        "p",
        "q_family",
    ]
    for column in required:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[required]


def construct_run_condition_table(behavior: pd.DataFrame) -> pd.DataFrame:
    required = {"subject", "run", "condition"}
    missing = sorted(required.difference(behavior.columns))
    if missing:
        raise ValueError(f"Behavior table is missing required columns: {missing}")

    data = behavior.copy()
    data["subject"] = data["subject"].map(canonical_subject)
    rows: list[dict[str, object]] = []

    for (subject, run), group in data.groupby(["subject", "run"], sort=True):
        condition_counts = group["condition"].astype(str).value_counts()
        if condition_counts.empty:
            continue
        rows.append(
            {
                "subject": subject,
                "run": int(run),
                "condition": condition_counts.index[0],
                "n_trials": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def analysis_set_mask(qc: pd.DataFrame, analysis_set: str) -> pd.Series:
    if "keep_main" not in qc.columns:
        raise ValueError("QC table is missing the keep_main column.")
    main = qc["keep_main"].astype(bool)
    if analysis_set == "main":
        return main
    if analysis_set == "fd_bad_run_exclude":
        excluded = (
            qc["sensitivity_exclude_candidate"].astype(bool)
            if "sensitivity_exclude_candidate" in qc.columns
            else pd.Series(False, index=qc.index)
        )
        return main & ~excluded
    raise ValueError(f"Unsupported analysis set: {analysis_set}")


def summarize_run_counts(run_qc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for analysis_set in ("main", "fd_bad_run_exclude"):
        subset = run_qc[analysis_set_mask(run_qc, analysis_set)]
        for condition, group in subset.groupby("condition", sort=True):
            rows.append(
                {
                    "analysis_set": analysis_set,
                    "condition": condition,
                    "n_runs": int(len(group)),
                    "n_subjects": int(group["subject"].nunique()),
                    "mean_fd": (
                        pd.to_numeric(group["fd_mean"], errors="coerce").mean()
                        if "fd_mean" in group.columns
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_contrast_counts(run_qc: pd.DataFrame) -> pd.DataFrame:
    contrast_conditions = {
        "up_minus_passive_FE": ["up", "passive"],
        "down_minus_passive_FE": ["down", "passive"],
        "up_minus_down_FE": ["up", "down"],
        "regulation_mean_minus_passive_FE": ["up", "down", "passive"],
        "stim_passive_FE": ["passive"],
    }
    rows: list[dict[str, object]] = []

    for analysis_set in ("main", "fd_bad_run_exclude"):
        subset = run_qc[analysis_set_mask(run_qc, analysis_set)]
        participant_conditions = {
            subject: set(group["condition"])
            for subject, group in subset.groupby("subject")
        }
        for contrast, required_conditions in contrast_conditions.items():
            count = sum(
                all(condition in available for condition in required_conditions)
                for available in participant_conditions.values()
            )
            rows.append(
                {
                    "analysis_set": analysis_set,
                    "contrast": contrast,
                    "n_subjects_with_required_runs": int(count),
                    "required_conditions": ",".join(required_conditions),
                }
            )
    return pd.DataFrame(rows)


def evidence_hierarchy() -> pd.DataFrame:
    rows = [
        [
            "Behavioral validation",
            "targeted reconstruction",
            "stimulus/condition effects",
            "small targeted family",
            "task validity",
        ],
        [
            "State-dependence",
            "qualified/exploratory",
            "direction × passive-expected pain",
            "sensitivity + coupling-null",
            "not a central mechanistic result",
        ],
        [
            "Down-regulation ROI attenuation",
            "targeted within-dataset result",
            "S2/operculum + insular complex",
            "family q + all-ROI q",
            "robustness assessed within this dataset",
        ],
        [
            "NPS attenuation",
            "targeted sensitivity result",
            "NPS extraction route",
            "signature family",
            "interpret with extraction-route sensitivity",
        ],
        [
            "dlPFC/IFJ→vmPFC lagged coupling",
            "exploratory/post-selection",
            "lagged regression vs up-regulation success",
            "screening q + matched controls",
            "temporal regression, not physiological causality",
        ],
        [
            "DCM",
            "model-dependent sensitivity",
            "focused DCM pathways",
            "model-specific",
            "no direct numerical convergence",
        ],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "claim_domain",
            "status",
            "test_family",
            "multiplicity_control",
            "allowed_interpretation",
        ],
    )


def write_interpretation(
    outdir: Path,
    roi_path: Path,
    behavior_path: Path,
    qc_path: Path,
    project: Path,
) -> None:
    text = f"""# ROI inference and quality-control tables

## Inputs

- ROI statistics: `{display_path(roi_path, project)}`
- Behavioral trial table: `{display_path(behavior_path, project)}`
- Run-level quality-control table: `{display_path(qc_path, project)}`

## Interpretation

Family-level and all-ROI corrected P values should be reported together. The
inferential hierarchy distinguishes within-dataset robustness from independent
replication and separates primary results from sensitivity and exploratory
analyses.
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--roi-stats", type=Path)
    parser.add_argument("--behavior-table", type=Path)
    parser.add_argument("--qc-table", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    roi_path = (
        args.roi_stats.expanduser().resolve()
        if args.roi_stats is not None
        else project / "results/roi_signatures/roi_stats.tsv"
    )
    behavior_path = (
        args.behavior_table.expanduser().resolve()
        if args.behavior_table is not None
        else project / "results/behavior/behavior_trials_master_clean.tsv.gz"
    )
    qc_path = (
        args.qc_table.expanduser().resolve()
        if args.qc_table is not None
        else project / "results/preprocessing_qc/analysis_inclusion_table.tsv"
    )
    for path in (roi_path, behavior_path, qc_path):
        if not path.exists():
            raise FileNotFoundError(path)

    roi_statistics = standardize_roi_statistics(roi_path)
    target = roi_statistics[
        roi_statistics["effect"].astype(str).str.contains(
            "down_minus_passive", case=False, na=False
        )
    ].copy()
    if not target.empty:
        target["q_all_roi_within_analysis_set"] = target.groupby(
            ["analysis_set", "effect"]
        )["p"].transform(benjamini_hochberg)
        sorted_target = target.sort_values(
            ["analysis_set", "q_all_roi_within_analysis_set", "p"]
        )
        sorted_target.to_csv(
            outdir / "full_32_roi_down_minus_passive_table.tsv",
            sep="\t",
            index=False,
        )
        target[
            [
                "analysis_set",
                "effect",
                "roi",
                "family",
                "mean",
                "stat",
                "p",
                "q_family",
                "q_all_roi_within_analysis_set",
            ]
        ].to_csv(outdir / "roi_all_roi_fdr_table.tsv", sep="\t", index=False)

    behavior = read_table(behavior_path)
    qc = read_table(qc_path)
    required_qc = {"subject", "run"}
    missing_qc = sorted(required_qc.difference(qc.columns))
    if missing_qc:
        raise ValueError(f"QC table is missing required columns: {missing_qc}")
    qc["subject"] = qc["subject"].map(canonical_subject)

    run_qc = construct_run_condition_table(behavior).merge(
        qc, on=["subject", "run"], how="left"
    )
    run_qc.to_csv(outdir / "run_condition_qc_table.tsv", sep="\t", index=False)
    summarize_run_counts(run_qc).to_csv(
        outdir / "condition_by_analysis_set_run_counts.tsv",
        sep="\t",
        index=False,
    )
    summarize_contrast_counts(run_qc).to_csv(
        outdir / "contrast_by_analysis_set_subject_counts.tsv",
        sep="\t",
        index=False,
    )
    evidence_hierarchy().to_csv(
        outdir / "inferential_hierarchy.tsv", sep="\t", index=False
    )
    write_interpretation(outdir, roi_path, behavior_path, qc_path, project)

    print(f"ROI inference and QC tables written to: {outdir}")


if __name__ == "__main__":
    main()
