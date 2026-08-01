#!/usr/bin/env python3
"""Run-level neural matched-passive sensitivity analysis.

The workflow aggregates trialwise ROI or signature estimates within each run,
then compares regulation runs with passive runs selected by run position. It
supports long-format tables with one feature per row and wide-format tables
whose feature columns begin with ``roi_``, ``nps_``, or ``signature_``.

This analysis is a sensitivity control for run-position and task-context
confounding. It does not replace the primary fixed-effect map analysis.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if path.name.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t", nrows=nrows)
    if path.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", nrows=nrows)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=nrows)
    raise ValueError(f"Unsupported table format: {path}")


def canonical_subject(value: object) -> str:
    text = str(value)
    match = re.search(r"sub[-_]?(\d+)", text, flags=re.I)
    if match:
        return f"sub-{int(match.group(1)):02d}"
    match = re.search(r"(\d+)", text)
    return f"sub-{int(match.group(1)):02d}" if match else text


def infer_column(
    frame: pd.DataFrame,
    exact: set[str],
    contains: tuple[str, ...] = (),
) -> str | None:
    for column in frame.columns:
        if normalize_name(column) in exact:
            return column
    for column in frame.columns:
        normalized = normalize_name(column)
        if any(token in normalized for token in contains):
            return column
    return None


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


def display_path(path: Path, project: Path) -> str:
    try:
        return path.resolve().relative_to(project.resolve()).as_posix()
    except ValueError:
        return path.name


def standardize_run_feature_table(
    frame: pd.DataFrame,
    run_condition_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert long- or wide-format input to a common run-feature table."""
    subject_column = infer_column(
        frame, {"subject", "participant", "sub"}, ("subject",)
    )
    run_column = infer_column(frame, {"run", "run_number"}, ("run",))
    condition_column = infer_column(
        frame, {"condition", "cond", "task_condition"}
    )
    feature_column = infer_column(
        frame, {"roi", "region", "name", "mask", "signature"}
    )
    value_column = infer_column(
        frame,
        {"roi_beta", "value", "estimate", "beta", "mean", "expression", "score"},
        ("roi_beta", "nps_weighted", "nps_dot", "signature"),
    )

    if subject_column is None or run_column is None:
        raise ValueError("Could not identify subject and run columns.")

    data = frame.copy()
    data[subject_column] = data[subject_column].map(canonical_subject)
    data[run_column] = pd.to_numeric(data[run_column], errors="coerce").astype(
        "Int64"
    )

    if condition_column is None and run_condition_table is not None:
        run_conditions = run_condition_table.copy()
        required = {"subject", "run", "condition"}
        missing = sorted(required.difference(run_conditions.columns))
        if missing:
            raise ValueError(
                f"Run-condition table is missing required columns: {missing}"
            )
        run_conditions["subject"] = run_conditions["subject"].map(
            canonical_subject
        )
        data = data.merge(
            run_conditions[["subject", "run", "condition"]],
            left_on=[subject_column, run_column],
            right_on=["subject", "run"],
            how="left",
        )
        condition_column = "condition"

    if feature_column is not None and value_column is not None:
        selected = [subject_column, run_column]
        if condition_column is not None:
            selected.append(condition_column)
        selected.extend([feature_column, value_column])
        output = data[selected].copy()

        rename = {
            subject_column: "subject",
            run_column: "run",
            feature_column: "feature",
            value_column: "value",
        }
        if condition_column is not None:
            rename[condition_column] = "condition"
        output = output.rename(columns=rename)
        if "condition" not in output.columns:
            output["condition"] = np.nan
    else:
        identifier_columns = [subject_column, run_column]
        if condition_column is not None:
            identifier_columns.append(condition_column)

        feature_columns: list[str] = []
        for column in data.columns:
            if column in identifier_columns:
                continue
            normalized = normalize_name(column)
            if not normalized.startswith(("roi_", "nps_", "signature_")):
                continue
            numeric = pd.to_numeric(data[column], errors="coerce")
            if numeric.notna().sum() >= max(5, int(0.2 * len(numeric))):
                data[column] = numeric
                feature_columns.append(column)

        if not feature_columns:
            raise ValueError(
                "No ROI or signature value columns were identified. "
                "Provide a long-format table with feature and value columns, "
                "or a wide-format table with roi_, nps_, or signature_ columns."
            )

        output = data.melt(
            id_vars=identifier_columns,
            value_vars=feature_columns,
            var_name="feature",
            value_name="value",
        )
        rename = {subject_column: "subject", run_column: "run"}
        if condition_column is not None:
            rename[condition_column] = "condition"
        output = output.rename(columns=rename)
        if "condition" not in output.columns:
            output["condition"] = np.nan

    output["feature"] = output["feature"].astype(str).str.replace(
        r"^roi_", "", regex=True
    )
    output["value"] = pd.to_numeric(output["value"], errors="coerce")
    output = output.dropna(subset=["subject", "run", "feature", "value"])
    return output[["subject", "run", "condition", "feature", "value"]]


def compute_matched_differences(
    run_features: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """Compute regulation-minus-passive differences for each feature."""
    required = {
        "subject",
        "reg_run",
        "reg_condition",
        "nearest_passive_run",
        "passive_before_run",
        "passive_after_run",
    }
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(f"Matched-run table is missing required columns: {missing}")

    lookup = run_features.groupby(["subject", "run", "feature"])["value"].mean()
    rows: list[dict[str, object]] = []

    for _, match in matches.iterrows():
        subject = canonical_subject(match["subject"])
        regulation_run = int(match["reg_run"])
        regulation_condition = match["reg_condition"]

        comparators: list[tuple[str, int | list[int]]] = []
        for column in (
            "nearest_passive_run",
            "passive_before_run",
            "passive_after_run",
        ):
            if pd.notna(match[column]):
                comparators.append((column.removesuffix("_run"), int(match[column])))

        if pd.notna(match["passive_before_run"]) and pd.notna(
            match["passive_after_run"]
        ):
            comparators.append(
                (
                    "adjacent_mean_passive",
                    [
                        int(match["passive_before_run"]),
                        int(match["passive_after_run"]),
                    ],
                )
            )

        features = run_features.loc[
            (run_features["subject"] == subject)
            & (run_features["run"] == regulation_run),
            "feature",
        ].unique()

        for feature in features:
            key = (subject, regulation_run, feature)
            if key not in lookup.index:
                continue
            regulation_value = float(lookup.loc[key])

            for comparator_name, passive_run in comparators:
                if isinstance(passive_run, list):
                    values = [
                        float(lookup.loc[(subject, run, feature)])
                        for run in passive_run
                        if (subject, run, feature) in lookup.index
                    ]
                    if not values:
                        continue
                    passive_value = float(np.mean(values))
                else:
                    passive_key = (subject, passive_run, feature)
                    if passive_key not in lookup.index:
                        continue
                    passive_value = float(lookup.loc[passive_key])

                rows.append(
                    {
                        "subject": subject,
                        "reg_condition": regulation_condition,
                        "reg_run": regulation_run,
                        "comparator": comparator_name,
                        "passive_run": str(passive_run),
                        "feature": feature,
                        "reg_value": regulation_value,
                        "passive_value": passive_value,
                        "diff_reg_minus_passive": regulation_value - passive_value,
                    }
                )

    return pd.DataFrame(rows)


def summarize_differences(differences: pd.DataFrame) -> pd.DataFrame:
    """Summarize matched differences and apply within-comparator FDR."""
    rows: list[dict[str, object]] = []
    grouped = differences.groupby(
        ["reg_condition", "comparator", "feature"], sort=True
    )

    for (condition, comparator, feature), group in grouped:
        values = pd.to_numeric(
            group["diff_reg_minus_passive"], errors="coerce"
        ).dropna()
        if len(values) < 5:
            continue
        statistic, p_value = stats.ttest_1samp(values, 0.0)
        standard_deviation = values.std(ddof=1)
        rows.append(
            {
                "reg_condition": condition,
                "comparator": comparator,
                "feature": feature,
                "n_subjects": int(len(values)),
                "mean_diff": values.mean(),
                "sd": standard_deviation,
                "se": standard_deviation / np.sqrt(len(values)),
                "t": statistic,
                "p": p_value,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary["q_all_features_within_condition_comparator"] = summary.groupby(
        ["reg_condition", "comparator"]
    )["p"].transform(benjamini_hochberg)
    return summary.sort_values(
        [
            "reg_condition",
            "comparator",
            "q_all_features_within_condition_comparator",
            "p",
        ]
    ).reset_index(drop=True)


def write_interpretation(
    outdir: Path,
    run_feature_path: Path,
    matched_run_path: Path,
    project: Path,
) -> None:
    text = f"""# Neural matched-passive ROI analysis

## Inputs

- Trialwise neural table: `{display_path(run_feature_path, project)}`
- Matched passive-run table: `{display_path(matched_run_path, project)}`

## Interpretation

Negative regulation-minus-passive differences indicate lower run-level neural
estimates during regulation than during the selected passive comparator. The
analysis is a run-position sensitivity control and should be interpreted
alongside, rather than in place of, the primary fixed-effect map results.
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--run-roi-table", type=Path, required=True)
    parser.add_argument("--matched-run-table", type=Path, required=True)
    parser.add_argument("--run-condition-table", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    run_feature_path = args.run_roi_table.expanduser().resolve()
    matched_run_path = args.matched_run_table.expanduser().resolve()
    if not run_feature_path.exists():
        raise FileNotFoundError(run_feature_path)
    if not matched_run_path.exists():
        raise FileNotFoundError(matched_run_path)

    run_condition_table = None
    if args.run_condition_table is not None:
        run_condition_path = args.run_condition_table.expanduser().resolve()
        if not run_condition_path.exists():
            raise FileNotFoundError(run_condition_path)
        run_condition_table = read_table(run_condition_path)

    run_features = standardize_run_feature_table(
        read_table(run_feature_path), run_condition_table
    )
    matches = read_table(matched_run_path)
    differences = compute_matched_differences(run_features, matches)
    summary = summarize_differences(differences)

    run_features.to_csv(
        outdir / "standardized_run_level_roi_signature_values.tsv",
        sep="\t",
        index=False,
    )
    differences.to_csv(
        outdir / "neural_matched_passive_subject_differences.tsv",
        sep="\t",
        index=False,
    )
    summary.to_csv(
        outdir / "neural_matched_passive_summary.tsv", sep="\t", index=False
    )
    write_interpretation(
        outdir, run_feature_path, matched_run_path, project
    )

    print(f"Neural matched-passive results written to: {outdir}")


if __name__ == "__main__":
    main()
