#!/usr/bin/env python3
"""Audit consistency among NPS contrast definitions.

The input table contains participant-level Neurologic Pain Signature (NPS)
values extracted from direct contrast maps and values reconstructed from
condition-level summaries. The workflow compares these analysis routes at the
participant level, quantifies agreement, and reports arithmetic condition
contrasts when the required condition columns are available.
"""

from __future__ import annotations

import argparse
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


def display_path(path: Path, project: Path) -> str:
    try:
        return path.resolve().relative_to(project.resolve()).as_posix()
    except ValueError:
        return path.name


def paired_test(difference: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(difference, errors="coerce").dropna()
    if len(values) < 3:
        return np.nan, np.nan
    statistic, p_value = stats.ttest_1samp(values, 0.0)
    return float(statistic), float(p_value)


def pearson_correlation(
    first: pd.Series, second: pd.Series
) -> tuple[float, float]:
    frame = pd.DataFrame(
        {
            "first": pd.to_numeric(first, errors="coerce"),
            "second": pd.to_numeric(second, errors="coerce"),
        }
    ).dropna()
    if len(frame) < 5 or frame["first"].std() == 0 or frame["second"].std() == 0:
        return np.nan, np.nan
    correlation, p_value = stats.pearsonr(frame["first"], frame["second"])
    return float(correlation), float(p_value)


def filter_analysis_rows(
    frame: pd.DataFrame,
    set_filter: str | None,
    measure_filter: str | None,
) -> pd.DataFrame:
    data = frame.copy()
    if set_filter is not None and "set_name" in data.columns:
        data = data[data["set_name"].astype(str) == set_filter]
    if measure_filter is not None and "measure" in data.columns:
        data = data[data["measure"].astype(str) == measure_filter]
    return data


def compare_direct_and_paired(
    frame: pd.DataFrame,
    direct_column: str,
    paired_column: str,
    contrast: str,
    set_filter: str | None,
    measure_filter: str | None,
) -> tuple[pd.DataFrame | None, dict[str, object] | None]:
    data = filter_analysis_rows(frame, set_filter, measure_filter)
    if direct_column not in data.columns or paired_column not in data.columns:
        return None, None

    identifier_columns = ["subject"]
    if "set_name" in data.columns:
        identifier_columns.append("set_name")
    if "measure" in data.columns:
        identifier_columns.append("measure")

    subject_level = data[
        identifier_columns + [direct_column, paired_column]
    ].copy()
    subject_level = subject_level.rename(
        columns={
            direct_column: "direct_contrast_nps",
            paired_column: "paired_condition_nps",
        }
    )
    subject_level["contrast"] = contrast
    subject_level["difference_direct_minus_paired"] = (
        subject_level["direct_contrast_nps"]
        - subject_level["paired_condition_nps"]
    )
    subject_level["same_sign"] = np.sign(
        subject_level["direct_contrast_nps"]
    ) == np.sign(subject_level["paired_condition_nps"])

    correlation, correlation_p = pearson_correlation(
        subject_level["direct_contrast_nps"],
        subject_level["paired_condition_nps"],
    )
    paired_t, paired_p = paired_test(
        subject_level["difference_direct_minus_paired"]
    )

    complete = subject_level[
        ["direct_contrast_nps", "paired_condition_nps"]
    ].dropna()
    summary = {
        "contrast": contrast,
        "direct_col": direct_column,
        "paired_col": paired_column,
        "set_filter": set_filter if set_filter is not None else "all",
        "measure_filter": measure_filter if measure_filter is not None else "all",
        "n_subjects": int(len(complete)),
        "direct_mean": pd.to_numeric(
            subject_level["direct_contrast_nps"], errors="coerce"
        ).mean(),
        "paired_mean": pd.to_numeric(
            subject_level["paired_condition_nps"], errors="coerce"
        ).mean(),
        "mean_difference_direct_minus_paired": pd.to_numeric(
            subject_level["difference_direct_minus_paired"], errors="coerce"
        ).mean(),
        "direct_positive_fraction": float(
            (
                pd.to_numeric(
                    subject_level["direct_contrast_nps"], errors="coerce"
                )
                > 0
            ).mean()
        ),
        "paired_positive_fraction": float(
            (
                pd.to_numeric(
                    subject_level["paired_condition_nps"], errors="coerce"
                )
                > 0
            ).mean()
        ),
        "same_sign_fraction": float(subject_level["same_sign"].mean()),
        "pearson_r": correlation,
        "pearson_p": correlation_p,
        "paired_t": paired_t,
        "paired_p": paired_p,
    }
    return subject_level, summary


def arithmetic_contrast_summary(
    frame: pd.DataFrame,
    set_filter: str | None,
    measure_filter: str | None,
) -> pd.DataFrame:
    data = filter_analysis_rows(frame, set_filter, measure_filter)
    required_conditions = {"cond_up", "cond_down", "cond_passive"}
    if not required_conditions.issubset(data.columns):
        return pd.DataFrame()

    arithmetic = {
        "up_minus_passive": data["cond_up"] - data["cond_passive"],
        "down_minus_passive": data["cond_down"] - data["cond_passive"],
        "regulation_mean_minus_passive": (
            (data["cond_up"] + data["cond_down"]) / 2.0
        )
        - data["cond_passive"],
        "up_minus_down": data["cond_up"] - data["cond_down"],
    }
    direct_columns = {
        "up_minus_passive": "up_minus_passive_FE",
        "down_minus_passive": "down_minus_passive_FE",
        "regulation_mean_minus_passive": "regulation_mean_minus_passive_FE",
        "up_minus_down": "up_minus_down_FE",
    }

    rows: list[dict[str, object]] = []
    for contrast, arithmetic_values in arithmetic.items():
        direct_column = direct_columns[contrast]
        if direct_column not in data.columns:
            continue

        correlation, correlation_p = pearson_correlation(
            data[direct_column], arithmetic_values
        )
        difference = data[direct_column] - arithmetic_values
        paired_t, paired_p = paired_test(difference)
        complete = pd.DataFrame(
            {"direct": data[direct_column], "arithmetic": arithmetic_values}
        ).dropna()
        rows.append(
            {
                "contrast": contrast,
                "direct_col": direct_column,
                "arithmetic_condition_difference": contrast,
                "n_subjects": int(len(complete)),
                "direct_mean": pd.to_numeric(
                    data[direct_column], errors="coerce"
                ).mean(),
                "arithmetic_mean": pd.to_numeric(
                    arithmetic_values, errors="coerce"
                ).mean(),
                "mean_difference_direct_minus_arithmetic": pd.to_numeric(
                    difference, errors="coerce"
                ).mean(),
                "pearson_r": correlation,
                "pearson_p": correlation_p,
                "paired_t": paired_t,
                "paired_p": paired_p,
                "same_sign_fraction": float(
                    (
                        np.sign(data[direct_column])
                        == np.sign(arithmetic_values)
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def write_interpretation(
    outdir: Path,
    table_path: Path,
    project: Path,
    set_filter: str | None,
    measure_filter: str | None,
) -> None:
    text = f"""# NPS contrast-consistency audit

## Input

- Participant-level NPS table: `{display_path(table_path, project)}`
- Analysis-set filter: `{set_filter if set_filter is not None else 'all'}`
- Measure filter: `{measure_filter if measure_filter is not None else 'all'}`

## Interpretation

The output tables compare NPS values obtained from direct contrast maps with
values reconstructed from condition-level summaries. Differences in sign or
magnitude indicate analysis-route sensitivity and should be reported when
interpreting NPS regulation effects.
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--nps-subject-condition-table", type=Path)
    parser.add_argument("--set-filter", default="main")
    parser.add_argument("--measure-filter")
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    table_path = (
        args.nps_subject_condition_table.expanduser().resolve()
        if args.nps_subject_condition_table is not None
        else project / "results/roi_signatures/nps_subject_condition_values.tsv"
    )
    if not table_path.exists():
        raise FileNotFoundError(table_path)

    frame = read_table(table_path)
    if "subject" not in frame.columns:
        raise ValueError("NPS table is missing the required subject column.")

    frame.to_csv(
        outdir / "nps_subject_condition_values_used.tsv", sep="\t", index=False
    )

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

    subject_tables: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for direct_column, paired_column, contrast in contrast_pairs:
        subject_table, summary = compare_direct_and_paired(
            frame,
            direct_column,
            paired_column,
            contrast,
            args.set_filter,
            args.measure_filter,
        )
        if subject_table is not None and summary is not None:
            subject_tables.append(subject_table)
            summaries.append(summary)

    subject_output = (
        pd.concat(subject_tables, ignore_index=True)
        if subject_tables
        else pd.DataFrame()
    )
    subject_output.to_csv(
        outdir / "nps_direct_vs_paired_subject_level.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(summaries).to_csv(
        outdir / "nps_direct_vs_paired_summary.tsv", sep="\t", index=False
    )
    arithmetic_contrast_summary(
        frame, args.set_filter, args.measure_filter
    ).to_csv(
        outdir / "nps_direct_vs_arithmetic_condition_difference_summary.tsv",
        sep="\t",
        index=False,
    )

    write_interpretation(
        outdir,
        table_path,
        project,
        args.set_filter,
        args.measure_filter,
    )
    print(f"NPS contrast-consistency results written to: {outdir}")


if __name__ == "__main__":
    main()
