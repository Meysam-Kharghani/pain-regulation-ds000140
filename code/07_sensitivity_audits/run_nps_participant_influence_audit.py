#!/usr/bin/env python3
"""Quantify participant influence on NPS contrast estimates.

The workflow evaluates a prespecified participant of interest, computes
leave-one-participant-out means for each NPS contrast, and lists the largest
standardized participant-level deviations. The default participant identifier
is ``sub-06`` because this participant was identified in the analysis audit;
an alternative identifier may be supplied with ``--participant-id``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


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


def paired_statistics(first: pd.Series, second: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "first": pd.to_numeric(first, errors="coerce"),
            "second": pd.to_numeric(second, errors="coerce"),
        }
    ).dropna()
    if len(frame) < 3:
        return {}

    difference = frame["first"] - frame["second"]
    paired_t, paired_p = stats.ttest_1samp(difference, 0.0)
    if (
        len(frame) >= 5
        and frame["first"].std() > 0
        and frame["second"].std() > 0
    ):
        correlation, correlation_p = stats.pearsonr(
            frame["first"], frame["second"]
        )
    else:
        correlation, correlation_p = np.nan, np.nan

    return {
        "n": int(len(frame)),
        "a_mean": frame["first"].mean(),
        "b_mean": frame["second"].mean(),
        "mean_difference": difference.mean(),
        "paired_t": paired_t,
        "paired_p": paired_p,
        "pearson_r": correlation,
        "pearson_p": correlation_p,
        "same_sign_fraction": float(
            (np.sign(frame["first"]) == np.sign(frame["second"])).mean()
        ),
    }


def run_influence_audit(
    frame: pd.DataFrame,
    participant_id: str,
    analysis_set: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    if "subject" not in data.columns:
        raise ValueError("NPS table is missing the required subject column.")
    data["subject"] = data["subject"].map(canonical_subject)

    if analysis_set is not None and "set_name" in data.columns:
        data = data[data["set_name"].astype(str) == analysis_set].copy()

    participant_id = canonical_subject(participant_id)
    contrast_pairs = [
        ("up_minus_passive_FE", "paired_up_minus_passive", "up_minus_passive"),
        (
            "down_minus_passive_FE",
            "paired_down_minus_passive",
            "down_minus_passive",
        ),
        (
            "regulation_mean_minus_passive_FE",
            "paired_regulation_mean_minus_passive",
            "regulation_mean_minus_passive",
        ),
        ("up_minus_down_FE", "paired_up_minus_down", "up_minus_down"),
    ]

    influence_rows: list[dict[str, object]] = []
    participant_rows: list[pd.DataFrame] = []
    leave_one_out_rows: list[dict[str, object]] = []
    deviation_tables: list[pd.DataFrame] = []

    measures = (
        sorted(data["measure"].dropna().unique())
        if "measure" in data.columns
        else ["all"]
    )

    for measure in measures:
        measure_data = data if measure == "all" else data[data["measure"] == measure]

        for direct_column, paired_column, contrast in contrast_pairs:
            if direct_column not in measure_data.columns:
                continue

            values = pd.to_numeric(measure_data[direct_column], errors="coerce")
            reference_mask = measure_data["subject"] != participant_id
            participant_mask = measure_data["subject"] == participant_id
            reference_values = values[reference_mask]
            participant_value = values[participant_mask].mean()
            reference_sd = reference_values.std(ddof=1)
            participant_z = (
                (participant_value - reference_values.mean()) / reference_sd
                if participant_mask.any()
                and np.isfinite(reference_sd)
                and reference_sd != 0
                else np.nan
            )

            row: dict[str, object] = {
                "measure": measure,
                "contrast": contrast,
                "direct_col": direct_column,
                "participant_id": participant_id,
                "n_all": int(values.notna().sum()),
                "direct_mean_all": values.mean(),
                "direct_mean_without_participant": reference_values.mean(),
                "participant_value": participant_value,
                "participant_z": participant_z,
                "direct_positive_fraction_all": float((values > 0).mean()),
                "direct_positive_fraction_without_participant": float(
                    (reference_values > 0).mean()
                ),
            }
            if paired_column in measure_data.columns:
                paired = paired_statistics(
                    measure_data[direct_column], measure_data[paired_column]
                )
                row.update(
                    {f"direct_vs_paired_{key}": value for key, value in paired.items()}
                )
            influence_rows.append(row)

            selected_columns = ["subject", direct_column]
            if paired_column in measure_data.columns:
                selected_columns.append(paired_column)
            participant_values = measure_data.loc[
                participant_mask, selected_columns
            ].copy()
            if not participant_values.empty:
                participant_values["measure"] = measure
                participant_values["contrast"] = contrast
                participant_values["participant_id"] = participant_id
                participant_rows.append(participant_values)

            for omitted_participant in measure_data["subject"].dropna().unique():
                retained = measure_data[
                    measure_data["subject"] != omitted_participant
                ]
                retained_values = pd.to_numeric(
                    retained[direct_column], errors="coerce"
                )
                leave_one_out_rows.append(
                    {
                        "measure": measure,
                        "contrast": contrast,
                        "omitted_participant": omitted_participant,
                        "n": int(retained_values.notna().sum()),
                        "direct_mean_without_participant": retained_values.mean(),
                        "direct_sd_without_participant": retained_values.std(ddof=1),
                    }
                )

            standard_deviation = values.std(ddof=1)
            standardized = (
                (values - values.mean()) / standard_deviation
                if np.isfinite(standard_deviation) and standard_deviation != 0
                else pd.Series(np.nan, index=values.index)
            )
            deviations = pd.DataFrame(
                {
                    "subject": measure_data["subject"],
                    "measure": measure,
                    "contrast": contrast,
                    "direct_col": direct_column,
                    "direct_value": values,
                    "z_score": standardized,
                }
            )
            deviations = deviations.sort_values(
                "z_score", key=lambda series: series.abs(), ascending=False
            ).head(10)
            deviation_tables.append(deviations)

    influence = pd.DataFrame(influence_rows)
    participant_values = (
        pd.concat(participant_rows, ignore_index=True)
        if participant_rows
        else pd.DataFrame()
    )
    leave_one_out = pd.DataFrame(leave_one_out_rows)
    largest_deviations = (
        pd.concat(deviation_tables, ignore_index=True)
        if deviation_tables
        else pd.DataFrame()
    )
    return influence, participant_values, leave_one_out, largest_deviations


def write_interpretation(
    outdir: Path,
    table_path: Path,
    project: Path,
    participant_id: str,
    analysis_set: str | None,
) -> None:
    text = f"""# NPS participant-influence audit

## Input

- Participant-level NPS table: `{display_path(table_path, project)}`
- Participant of interest: `{canonical_subject(participant_id)}`
- Analysis-set filter: `{analysis_set if analysis_set is not None else 'all'}`

## Interpretation

The tables quantify how the participant of interest and each leave-one-
participant-out omission affect direct NPS contrast estimates. These results
should be used to describe participant influence and analysis sensitivity;
they do not provide a basis for excluding a participant without an independent
quality-control criterion.
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--nps-table", type=Path)
    parser.add_argument("--participant-id", default="sub-06")
    parser.add_argument("--analysis-set", default="main")
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    table_path = (
        args.nps_table.expanduser().resolve()
        if args.nps_table is not None
        else project / "results/roi_signatures/nps_subject_condition_values.tsv"
    )
    if not table_path.exists():
        raise FileNotFoundError(table_path)

    influence, participant_values, leave_one_out, largest_deviations = (
        run_influence_audit(
            read_table(table_path), args.participant_id, args.analysis_set
        )
    )

    influence.to_csv(
        outdir / "nps_participant_influence_summary.tsv", sep="\t", index=False
    )
    participant_values.to_csv(
        outdir / "nps_participant_values.tsv", sep="\t", index=False
    )
    leave_one_out.to_csv(
        outdir / "nps_leave_one_participant_out_means.tsv",
        sep="\t",
        index=False,
    )
    largest_deviations.to_csv(
        outdir / "nps_largest_participant_deviations.tsv",
        sep="\t",
        index=False,
    )
    write_interpretation(
        outdir,
        table_path,
        project,
        args.participant_id,
        args.analysis_set,
    )

    print(f"NPS participant-influence results written to: {outdir}")


if __name__ == "__main__":
    main()
