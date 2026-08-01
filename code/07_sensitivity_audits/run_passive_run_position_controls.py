#!/usr/bin/env python3
"""Behavioral controls for passive-run position.

For each up- or down-regulation run, the workflow identifies the nearest
passive run and, when available, the passive runs immediately before and after
the regulation run. It then computes participant-level matched differences in
pain ratings and estimates linear change across passive-run position.

The script operates on compact tables distributed with the repository. Input
paths may be supplied explicitly; otherwise repository-relative defaults are
used.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def read_table(path: Path) -> pd.DataFrame:
    """Read a tabular input from TSV, compressed TSV, or CSV."""
    if path.name.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def canonical_subject(value: object) -> str:
    """Return a BIDS-style participant identifier when possible."""
    text = str(value)
    match = re.search(r"sub[-_]?(\d+)", text, flags=re.I)
    if match:
        return f"sub-{int(match.group(1)):02d}"
    match = re.search(r"(\d+)", text)
    return f"sub-{int(match.group(1)):02d}" if match else text


def repository_path(project: Path, supplied: Path | None, relative: str) -> Path:
    """Resolve an explicit path or a repository-relative default."""
    path = supplied if supplied is not None else project / relative
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def display_path(path: Path, project: Path) -> str:
    """Return a portable path for reports without exposing local directories."""
    try:
        return path.resolve().relative_to(project.resolve()).as_posix()
    except ValueError:
        return path.name


def infer_run_condition(behavior: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial-level behavior to one row per participant and run."""
    required = {"subject", "run", "condition", "rating_raw"}
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
                "mean_rating_raw": pd.to_numeric(
                    group["rating_raw"], errors="coerce"
                ).mean(),
                "mean_rating_z_subject": (
                    pd.to_numeric(group["rating_z_subject"], errors="coerce").mean()
                    if "rating_z_subject" in group.columns
                    else np.nan
                ),
                "mean_temperature_raw": (
                    pd.to_numeric(group["temperature_raw"], errors="coerce").mean()
                    if "temperature_raw" in group.columns
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["subject", "run"]).reset_index(drop=True)


def identify_passive_comparators(run_table: pd.DataFrame) -> pd.DataFrame:
    """Identify nearest, preceding, and following passive runs."""
    rows: list[dict[str, object]] = []

    for subject, group in run_table.groupby("subject", sort=True):
        passive_runs = sorted(
            group.loc[group["condition"] == "passive", "run"].astype(int).tolist()
        )
        regulation = group[group["condition"].isin(["up", "down"])]

        for _, row in regulation.iterrows():
            regulation_run = int(row["run"])
            preceding = [run for run in passive_runs if run < regulation_run]
            following = [run for run in passive_runs if run > regulation_run]
            nearest = (
                sorted(passive_runs, key=lambda run: (abs(run - regulation_run), run))[0]
                if passive_runs
                else np.nan
            )
            rows.append(
                {
                    "subject": subject,
                    "reg_run": regulation_run,
                    "reg_condition": row["condition"],
                    "nearest_passive_run": nearest,
                    "passive_before_run": max(preceding) if preceding else np.nan,
                    "passive_after_run": min(following) if following else np.nan,
                }
            )

    return pd.DataFrame(rows)


def compute_matched_behavior(
    behavior: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """Compute participant-level rating differences for each comparator."""
    data = behavior.copy()
    data["subject"] = data["subject"].map(canonical_subject)
    rows: list[dict[str, object]] = []

    for _, match in matches.iterrows():
        for comparator_column in (
            "nearest_passive_run",
            "passive_before_run",
            "passive_after_run",
        ):
            if pd.isna(match[comparator_column]):
                continue

            regulation_run = int(match["reg_run"])
            passive_run = int(match[comparator_column])
            regulation = data[
                (data["subject"] == match["subject"])
                & (data["run"] == regulation_run)
            ]
            passive = data[
                (data["subject"] == match["subject"])
                & (data["run"] == passive_run)
            ]
            if regulation.empty or passive.empty:
                continue

            regulation_raw = pd.to_numeric(
                regulation["rating_raw"], errors="coerce"
            )
            passive_raw = pd.to_numeric(passive["rating_raw"], errors="coerce")

            if "rating_z_subject" in data.columns:
                regulation_z = pd.to_numeric(
                    regulation["rating_z_subject"], errors="coerce"
                )
                passive_z = pd.to_numeric(
                    passive["rating_z_subject"], errors="coerce"
                )
                matched_diff_z = regulation_z.mean() - passive_z.mean()
            else:
                matched_diff_z = np.nan

            rows.append(
                {
                    "subject": match["subject"],
                    "reg_condition": match["reg_condition"],
                    "reg_run": regulation_run,
                    "passive_run": passive_run,
                    "comparator": comparator_column.removesuffix("_run"),
                    "matched_diff_raw": regulation_raw.mean() - passive_raw.mean(),
                    "matched_diff_z": matched_diff_z,
                    "reg_mean_raw": regulation_raw.mean(),
                    "passive_mean_raw": passive_raw.mean(),
                    "run_distance": abs(regulation_run - passive_run),
                }
            )

    return pd.DataFrame(rows)


def summarize_matched_differences(matched: pd.DataFrame) -> pd.DataFrame:
    """Run one-sample tests of matched differences against zero."""
    rows: list[dict[str, object]] = []
    for (condition, comparator), group in matched.groupby(
        ["reg_condition", "comparator"], sort=True
    ):
        for metric in ("matched_diff_raw", "matched_diff_z"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(values) < 3:
                continue
            statistic, p_value = stats.ttest_1samp(values, 0.0)
            standard_deviation = values.std(ddof=1)
            rows.append(
                {
                    "reg_condition": condition,
                    "comparator": comparator,
                    "metric": metric,
                    "n_subjects": int(len(values)),
                    "mean": values.mean(),
                    "sd": standard_deviation,
                    "se": standard_deviation / np.sqrt(len(values)),
                    "t": statistic,
                    "p": p_value,
                }
            )
    return pd.DataFrame(rows)


def estimate_passive_run_trend(run_table: pd.DataFrame) -> pd.DataFrame:
    """Estimate passive-run position effects with participant fixed effects."""
    passive = run_table[run_table["condition"] == "passive"].copy()
    rows: list[dict[str, object]] = []

    for metric in ("mean_rating_raw", "mean_rating_z_subject"):
        if metric not in passive.columns or passive[metric].notna().sum() < 3:
            continue
        fit = smf.ols(f"{metric} ~ run + C(subject)", data=passive).fit()
        rows.append(
            {
                "metric": metric,
                "n_runs": int(fit.nobs),
                "run_slope": fit.params.get("run", np.nan),
                "run_se": fit.bse.get("run", np.nan),
                "run_t": fit.tvalues.get("run", np.nan),
                "run_p": fit.pvalues.get("run", np.nan),
                "r2": fit.rsquared,
            }
        )

    return pd.DataFrame(rows)


def write_interpretation(
    outdir: Path, behavior_path: Path, qc_path: Path, project: Path
) -> None:
    """Write a portable description of inputs and interpretation."""
    text = f"""# Passive run-position controls

## Inputs

- Behavioral trial table: `{display_path(behavior_path, project)}`
- Run-level quality-control table: `{display_path(qc_path, project)}`

## Interpretation

The matched tables quantify behavioral differences relative to passive runs at
comparable positions in the run sequence. These analyses are sensitivity
controls for run-position and habituation effects; they do not replace the
primary condition-level analysis.
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--behavior-table", type=Path)
    parser.add_argument("--qc-table", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    behavior_path = repository_path(
        project,
        args.behavior_table,
        "results/behavior/behavior_trials_master_clean.tsv.gz",
    )
    qc_path = repository_path(
        project,
        args.qc_table,
        "results/preprocessing_qc/analysis_inclusion_table.tsv",
    )

    behavior = read_table(behavior_path)
    run_table = infer_run_condition(behavior)

    qc = read_table(qc_path)
    required_qc = {"subject", "run"}
    missing_qc = sorted(required_qc.difference(qc.columns))
    if missing_qc:
        raise ValueError(f"QC table is missing required columns: {missing_qc}")
    qc["subject"] = qc["subject"].map(canonical_subject)
    retained_qc_columns = [
        "subject",
        "run",
        *[
            column
            for column in (
                "keep_main",
                "sensitivity_exclude_candidate",
                "analysis_decision",
                "fd_mean",
                "pct_fd_gt_0p5",
                "estimated_dof_after_nuisance",
            )
            if column in qc.columns
        ],
    ]
    run_table = run_table.merge(
        qc[retained_qc_columns], on=["subject", "run"], how="left"
    )

    matches = identify_passive_comparators(run_table)
    matched_behavior = compute_matched_behavior(behavior, matches)

    run_table.to_csv(outdir / "run_condition_table.tsv", sep="\t", index=False)
    matches.to_csv(outdir / "matched_passive_run_table.tsv", sep="\t", index=False)
    matched_behavior.to_csv(
        outdir / "matched_passive_behavior_subject_runs.tsv", sep="\t", index=False
    )
    summarize_matched_differences(matched_behavior).to_csv(
        outdir / "behavioral_run_position_matched_results.tsv",
        sep="\t",
        index=False,
    )
    estimate_passive_run_trend(run_table).to_csv(
        outdir / "run_position_habituation_summaries.tsv", sep="\t", index=False
    )
    write_interpretation(outdir, behavior_path, qc_path, project)

    print(f"Passive run-position controls written to: {outdir}")


if __name__ == "__main__":
    main()
