#!/usr/bin/env python3
"""
Prepare a clean, analysis-ready behavioral master table for ds000140.

Input:
    trial_table_enriched_standardized.tsv

Outputs:
    behavior_trials_master_clean.tsv
    behavior_trials_master_clean_qc.json
    behavior_trials_data_dictionary.tsv

Design principle:
    This script keeps only raw/enriched trial information and clean derived covariates.
    It intentionally does not carry over residuals, passive predictions, or success metrics.
    Those should be generated later by the modeling pipeline, so model outputs do not leak
    into model inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "subject", "run", "condition", "temperature",
    "stim_onset", "stim_onset_runrel", "rating_onset", "rating_onset_runrel",
    "rating_raw", "trial_in_run", "global_trial_index",
    "sex", "age", "stim_duration", "rating_duration", "trial_id",
]

MODEL_OUTPUT_PATTERNS = (
    "passive_pred", "residual", "success", "predicted", "fitted"
)

CONDITION_ORDER = ["passive", "up", "down"]


def zscore(s: pd.Series) -> pd.Series:
    """Z-score with ddof=0; returns NaN if variance is zero."""
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def minmax01(s: pd.Series) -> pd.Series:
    """Subject-wise 0-1 scaling; returns NaN if range is zero."""
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.nan, index=s.index)
    return (s - mn) / (mx - mn)


def first_nonmissing_row_value(df: pd.DataFrame, cols: List[str], default=np.nan):
    """Return first available column value row-wise from cols."""
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.Series(default, index=df.index)
    out = df[available[0]].copy()
    for c in available[1:]:
        out = out.where(out.notna(), df[c])
    return out


def ensure_required(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize categorical labels.
    df["subject"] = df["subject"].astype(str)
    df["condition"] = df["condition"].astype(str).str.strip().str.lower()
    bad_conditions = sorted(set(df["condition"].dropna()) - set(CONDITION_ORDER))
    if bad_conditions:
        raise ValueError(f"Unexpected condition labels: {bad_conditions}")

    # Convert numeric fields.
    numeric_cols = [
        "run", "temperature", "stim_onset", "stim_onset_runrel", "rating_onset",
        "rating_onset_runrel", "rating_raw", "trial_in_run", "global_trial_index",
        "age", "stim_duration", "rating_duration"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Apply stable sorting to establish the canonical trial order.
    sort_cols = ["subject", "global_trial_index", "run", "trial_in_run", "stim_onset_runrel"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def build_master(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_basic_columns(df)
    m = pd.DataFrame(index=df.index)

    # ---- Identifiers ----
    m["subject"] = df["subject"]
    m["sex"] = df.get("sex", pd.Series(pd.NA, index=df.index))
    m["age"] = df.get("age", pd.Series(np.nan, index=df.index))
    m["run"] = df["run"].astype("Int64")
    m["run_index"] = pd.to_numeric(df.get("run_index", df["run"]), errors="coerce").astype("Int64")
    m["trial_in_run"] = df["trial_in_run"].astype("Int64")
    m["global_trial_index"] = df["global_trial_index"].astype("Int64")
    m["trial_id"] = df["trial_id"].astype(str)
    if "events_file" in df.columns:
        m["events_file"] = df["events_file"].astype(str)

    # ---- Condition / task design ----
    m["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=False)
    m["is_reg_trial"] = df["condition"].isin(["up", "down"]).astype(int)

    # Regulation runs are expected to be run 3 and run 7, but use existing info when available.
    if "run_is_reg" in df.columns:
        m["is_reg_run"] = pd.to_numeric(df["run_is_reg"], errors="coerce").fillna(df["run"].isin([3, 7]).astype(int)).astype(int)
    elif "is_reg_run" in df.columns:
        m["is_reg_run"] = pd.to_numeric(df["is_reg_run"], errors="coerce").fillna(df["run"].isin([3, 7]).astype(int)).astype(int)
    else:
        m["is_reg_run"] = df["run"].isin([3, 7]).astype(int)

    m["reg_condition"] = df["condition"].where(df["condition"].isin(["up", "down"]), pd.NA)
    m["condition_code_ud"] = df["condition"].map({"up": 1.0, "down": -1.0, "passive": np.nan})
    m["condition_code_all"] = df["condition"].map({"up": 1.0, "passive": 0.0, "down": -1.0})
    m["condition_up"] = (df["condition"] == "up").astype(int)
    m["condition_down"] = (df["condition"] == "down").astype(int)
    m["condition_passive"] = (df["condition"] == "passive").astype(int)

    m["run3_condition"] = first_nonmissing_row_value(df, ["cond_run3", "run3_type"])
    m["run7_condition"] = first_nonmissing_row_value(df, ["cond_run7", "run7_type"])
    m["up_first"] = pd.to_numeric(first_nonmissing_row_value(df, ["up_first", "order_up_first"]), errors="coerce")
    m["up_first"] = m["up_first"].astype("Int64")
    m["order_label"] = np.where(m["up_first"] == 1, "up_first", np.where(m["up_first"] == 0, "down_first", pd.NA))

    # Passive context: run 5 and 6 were the shifted-temperature passive context in the paper.
    m["passive_context_shifted"] = ((df["condition"] == "passive") & (df["run"].isin([5, 6]))).astype(int)

    # ---- Stimulus / response variables ----
    m["temperature_raw"] = df["temperature"].astype(float)
    m["temperature_c_global"] = m["temperature_raw"] - m["temperature_raw"].mean()
    m["temperature_z_global"] = zscore(m["temperature_raw"])
    m["temperature_c_subject"] = m["temperature_raw"] - m.groupby("subject")["temperature_raw"].transform("mean")
    m["temperature_z_subject"] = m.groupby("subject", group_keys=False)["temperature_raw"].apply(zscore)

    m["rating_raw"] = df["rating_raw"].astype(float)
    m["rating_c_global"] = m["rating_raw"] - m["rating_raw"].mean()
    m["rating_z_global"] = zscore(m["rating_raw"])
    m["rating_c_subject"] = m["rating_raw"] - m.groupby("subject")["rating_raw"].transform("mean")
    m["rating_z_subject"] = m.groupby("subject", group_keys=False)["rating_raw"].apply(zscore)
    m["rating_01_subject"] = m.groupby("subject", group_keys=False)["rating_raw"].apply(minmax01)
    m["rating_over_100"] = (m["rating_raw"] > 100).astype(int)

    # Keep previous rating_z for comparison if present; do not use as canonical unless identical.
    if "rating_z" in df.columns:
        m["rating_z_existing"] = pd.to_numeric(df["rating_z"], errors="coerce")
        m["rating_z_diff_existing_minus_recomputed"] = m["rating_z_existing"] - m["rating_z_subject"]

    # ---- Timing ----
    m["stim_onset_abs"] = df["stim_onset"].astype(float)
    m["stim_onset_runrel"] = df["stim_onset_runrel"].astype(float)
    m["rating_onset_abs"] = df["rating_onset"].astype(float)

    # The legacy rating_onset_runrel column does not represent run-relative onset.
    # In this table it is relative to the first stimulus onset of the run.
    # Preserve it with a descriptive name and compute the run-relative onset.
    m["rating_onset_rel_first_stim_existing"] = df["rating_onset_runrel"].astype(float)
    m["rating_delay_from_stim_onset"] = m["rating_onset_abs"] - m["stim_onset_abs"]

    # Compute run-relative rating onset when absolute onset is cumulative across runs.
    m["run_start_abs_est"] = m["stim_onset_abs"] - m["stim_onset_runrel"]
    m["rating_onset_runrel"] = m["stim_onset_runrel"] + m["rating_delay_from_stim_onset"]
    m["first_stim_onset_runrel"] = m.groupby(["subject", "run"])["stim_onset_runrel"].transform("min")
    m["rating_onset_rel_first_stim_recomputed"] = m["rating_onset_runrel"] - m["first_stim_onset_runrel"]
    m["stim_duration"] = df["stim_duration"].astype(float)
    m["rating_duration"] = df["rating_duration"].astype(float)
    m["stim_offset_runrel"] = m["stim_onset_runrel"] + m["stim_duration"]
    m["rating_offset_runrel"] = m["rating_onset_runrel"] + m["rating_duration"]

    # ---- Centered time/order covariates ----
    m["run_centered"] = m["run"].astype(float) - m["run"].astype(float).mean()
    m["run_centered_subject"] = m["run"].astype(float) - m.groupby("subject")["run"].transform("mean").astype(float)
    m["trial_in_run_c"] = m["trial_in_run"].astype(float) - m.groupby(["subject", "run"])["trial_in_run"].transform("mean").astype(float)
    m["global_trial_z_subject"] = m.groupby("subject", group_keys=False)["global_trial_index"].apply(zscore)

    # ---- History variables: recomputed from canonical order ----
    gsub = m.groupby("subject", group_keys=False)
    m["prev_temperature_subject"] = gsub["temperature_raw"].shift(1)
    m["prev_rating_raw_subject"] = gsub["rating_raw"].shift(1)
    m["prev_rating_z_subject"] = gsub["rating_z_subject"].shift(1)
    m["prev_condition_subject"] = gsub["condition"].shift(1).astype("object")
    m["prev_run_subject"] = gsub["run"].shift(1).astype("Int64")
    m["has_prev_subject"] = m["prev_temperature_subject"].notna().astype(int)
    m["same_run_as_prev"] = ((m["run"] == m["prev_run_subject"]) & (m["has_prev_subject"] == 1)).astype(int)

    grun = m.groupby(["subject", "run"], group_keys=False)
    m["prev_temperature_withinrun"] = grun["temperature_raw"].shift(1)
    m["prev_rating_raw_withinrun"] = grun["rating_raw"].shift(1)
    m["prev_condition_withinrun"] = grun["condition"].shift(1).astype("object")
    m["prev_stim_onset_runrel_withinrun"] = grun["stim_onset_runrel"].shift(1)
    m["delta_prev_stim_withinrun"] = m["stim_onset_runrel"] - m["prev_stim_onset_runrel_withinrun"]
    m["has_prev_withinrun"] = m["prev_temperature_withinrun"].notna().astype(int)

    # History deviations, useful for computational models.
    m["prev_rating_minus_expected_subject_mean"] = m["prev_rating_raw_subject"] - gsub["rating_raw"].transform("mean")
    m["prev_temp_minus_subject_mean"] = m["prev_temperature_subject"] - gsub["temperature_raw"].transform("mean")

    # ---- Missingness / flags ----
    m["rating_missing"] = m["rating_raw"].isna().astype(int)
    m["temperature_missing"] = m["temperature_raw"].isna().astype(int)
    m["timing_missing"] = m[["stim_onset_runrel", "rating_onset_runrel", "stim_duration", "rating_duration"]].isna().any(axis=1).astype(int)

    # Verify that model-output columns are excluded from analysis inputs.
    leaked = [c for c in m.columns if any(pat in c.lower() for pat in MODEL_OUTPUT_PATTERNS)]
    if leaked:
        raise RuntimeError(f"Model-output-like columns leaked into master: {leaked}")

    return m.reset_index(drop=True)


def make_qc(df_raw: pd.DataFrame, master: pd.DataFrame) -> Dict:
    qc: Dict = {}
    qc["n_rows_raw"] = int(df_raw.shape[0])
    qc["n_cols_raw"] = int(df_raw.shape[1])
    qc["n_rows_master"] = int(master.shape[0])
    qc["n_cols_master"] = int(master.shape[1])
    qc["n_subjects"] = int(master["subject"].nunique())
    qc["subjects"] = sorted(master["subject"].unique().tolist())
    qc["condition_counts"] = master["condition"].astype(str).value_counts(dropna=False).to_dict()
    qc["run_counts"] = {str(k): int(v) for k, v in master["run"].value_counts().sort_index().items()}
    qc["trials_per_subject_min"] = int(master.groupby("subject").size().min())
    qc["trials_per_subject_max"] = int(master.groupby("subject").size().max())
    qc["duplicate_trial_id_count"] = int(master["trial_id"].duplicated().sum())
    qc["missing_by_column"] = {k: int(v) for k, v in master.isna().sum().items() if int(v) > 0}

    qc["rating_raw_min"] = float(master["rating_raw"].min())
    qc["rating_raw_max"] = float(master["rating_raw"].max())
    qc["rating_raw_over_100_count"] = int((master["rating_raw"] > 100).sum())
    qc["rating_raw_over_100_by_condition"] = {
        str(k): int(v) for k, v in master.groupby("condition", observed=False)["rating_over_100"].sum().items()
    }

    qc["temperature_unique_values"] = sorted([float(x) for x in master["temperature_raw"].dropna().unique()])
    qc["temperature_by_condition_counts"] = (
        master.groupby(["condition", "temperature_raw"], observed=False).size()
        .reset_index(name="n").to_dict(orient="records")
    )

    # Timing QC
    rel_first_diff = master["rating_onset_rel_first_stim_existing"] - master["rating_onset_rel_first_stim_recomputed"]
    qc["rating_onset_rel_first_stim_existing_minus_recomputed_abs_max"] = float(np.nanmax(np.abs(rel_first_diff)))
    old_vs_fixed = master["rating_onset_rel_first_stim_existing"] - master["rating_onset_runrel"]
    qc["rating_onset_runrel_existing_vs_recomputed_abs_median_diff"] = float(np.nanmedian(np.abs(old_vs_fixed)))
    run_start_var = master.groupby(["subject", "run"])["run_start_abs_est"].agg(lambda x: float(np.nanmax(x) - np.nanmin(x)))
    qc["run_start_abs_est_range_max_within_subject_run"] = float(run_start_var.max())
    qc["run_start_abs_est_range_median_within_subject_run"] = float(run_start_var.median())

    # Check balanced up/down temperature design.
    reg = master[master["condition"].isin(["up", "down"])].copy()
    if not reg.empty:
        temp_bal = reg.groupby(["condition", "temperature_raw"], observed=False).size().unstack("condition", fill_value=0)
        if "up" in temp_bal.columns and "down" in temp_bal.columns:
            qc["reg_up_down_temperature_count_abs_diff_sum"] = int((temp_bal["up"] - temp_bal["down"]).abs().sum())
            qc["reg_up_down_temperature_count_table"] = temp_bal.reset_index().to_dict(orient="records")

    # Existing z comparison
    if "rating_z_diff_existing_minus_recomputed" in master.columns:
        z_diff = master["rating_z_diff_existing_minus_recomputed"]
        qc["rating_z_existing_minus_recomputed_abs_max"] = float(np.nanmax(np.abs(z_diff)))
        qc["rating_z_existing_minus_recomputed_abs_median"] = float(np.nanmedian(np.abs(z_diff)))

    # Potential design consistency warnings.
    warnings = []
    if qc["duplicate_trial_id_count"] > 0:
        warnings.append("Duplicate trial_id values found.")
    if qc["run_start_abs_est_range_max_within_subject_run"] > 1e-3:
        warnings.append("run_start_abs_est is not constant within at least one subject/run; inspect timing columns.")
    if qc["rating_onset_rel_first_stim_existing_minus_recomputed_abs_max"] > 1e-3:
        warnings.append("Existing rating_onset_runrel does not match corrected rating_onset_runrel minus first stimulus onset; inspect timing columns.")
    if qc.get("reg_up_down_temperature_count_abs_diff_sum", 0) != 0:
        warnings.append("Regulation up/down trials are not perfectly balanced across recorded temperature values.")
    if qc["rating_raw_over_100_count"] > 0:
        warnings.append("rating_raw contains values >100; treat it as raw recorded scale, not necessarily a 0-100 VAS.")
    qc["warnings"] = warnings

    return qc


def make_dictionary() -> pd.DataFrame:
    rows = [
        ("subject", "Participant identifier."),
        ("sex", "Participant sex label as available in source table."),
        ("age", "Participant age as available in source table."),
        ("run", "Functional run number."),
        ("trial_in_run", "Trial index within run."),
        ("global_trial_index", "Trial index across the whole experiment within subject."),
        ("trial_id", "Unique trial identifier: subject/run/trial."),
        ("condition", "passive/up/down condition label."),
        ("is_reg_trial", "1 for up/down trials, 0 for passive."),
        ("is_reg_run", "1 for regulation runs, expected run 3 and run 7."),
        ("reg_condition", "up/down for regulation trials; missing for passive."),
        ("condition_code_ud", "up=+1, down=-1, passive=NaN; use only for up-vs-down models."),
        ("condition_code_all", "up=+1, passive=0, down=-1; use carefully as an ordered coding."),
        ("condition_up/down/passive", "Dummy-coded condition indicators."),
        ("run3_condition", "Which regulation condition was assigned to run 3."),
        ("run7_condition", "Which regulation condition was assigned to run 7."),
        ("up_first", "1 if run 3 was up and run 7 down; 0 if down first; missing if unknown."),
        ("order_label", "up_first/down_first."),
        ("passive_context_shifted", "1 for passive runs 5/6, which had shifted temperature context."),
        ("temperature_raw", "Recorded stimulus intensity/temperature value from source table."),
        ("temperature_c_global", "temperature_raw centered at grand mean."),
        ("temperature_z_global", "temperature_raw z-scored globally."),
        ("temperature_c_subject", "temperature_raw centered within subject."),
        ("temperature_z_subject", "temperature_raw z-scored within subject."),
        ("rating_raw", "Raw recorded pain/warmth rating."),
        ("rating_c_global", "rating_raw centered at grand mean."),
        ("rating_z_global", "rating_raw z-scored globally."),
        ("rating_c_subject", "rating_raw centered within subject."),
        ("rating_z_subject", "rating_raw z-scored within subject; recommended standardized rating."),
        ("rating_01_subject", "rating_raw min-max scaled to 0-1 within subject."),
        ("rating_over_100", "1 if rating_raw > 100."),
        ("stim_onset_abs", "Original stimulus onset from source table; may be cumulative across runs."),
        ("stim_onset_runrel", "Stimulus onset relative to current run."),
        ("rating_onset_abs", "Original rating onset from source table; may be cumulative across runs."),
        ("rating_onset_rel_first_stim_existing", "Original column named rating_onset_runrel; appears relative to the first stimulus onset of each run, not true run-relative time."),
        ("rating_delay_from_stim_onset", "rating_onset_abs - stim_onset_abs."),
        ("run_start_abs_est", "Estimated absolute run start: stim_onset_abs - stim_onset_runrel."),
        ("rating_onset_runrel", "Corrected rating onset relative to current run."),
        ("first_stim_onset_runrel", "First stimulus onset within subject/run."),
        ("rating_onset_rel_first_stim_recomputed", "Corrected rating_onset_runrel minus first_stim_onset_runrel; used to validate the existing timing column."),
        ("stim_duration/rating_duration", "Durations in seconds."),
        ("stim_offset_runrel/rating_offset_runrel", "Run-relative offsets."),
        ("run_centered", "Run centered at global mean."),
        ("run_centered_subject", "Run centered within subject."),
        ("trial_in_run_c", "Trial index centered within each subject/run."),
        ("global_trial_z_subject", "Global trial index z-scored within subject."),
        ("prev_*_subject", "Previous-trial variables across the subject's full experiment."),
        ("prev_*_withinrun", "Previous-trial variables within the same run."),
        ("has_prev_subject", "1 if a previous trial exists for that subject."),
        ("has_prev_withinrun", "1 if a previous trial exists in the same run."),
        ("delta_prev_stim_withinrun", "Difference from previous stimulus onset within the same run."),
        ("rating_missing/temperature_missing/timing_missing", "Basic missingness flags."),
    ]
    return pd.DataFrame(rows, columns=["column", "description"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean behavioral master table for ds000140.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("trial_table_enriched_standardized.tsv"),
        help="Path to enriched trial TSV. Use the enriched table, not the residuals table.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("behavior_master_clean"),
        help="Output directory.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.input, sep="\t")
    ensure_required(df_raw)

    # Reject residual or model-output tables when supplied as analysis inputs.
    input_model_cols = [c for c in df_raw.columns if any(pat in c.lower() for pat in MODEL_OUTPUT_PATTERNS)]
    if input_model_cols:
        raise ValueError(
            "Input table contains model-output-like columns. Use the enriched/raw trial table instead. "
            f"Problem columns: {input_model_cols}"
        )

    master = build_master(df_raw)
    qc = make_qc(df_raw, master)
    dictionary = make_dictionary()

    master_path = args.outdir / "behavior_trials_master_clean.tsv"
    qc_path = args.outdir / "behavior_trials_master_clean_qc.json"
    dict_path = args.outdir / "behavior_trials_data_dictionary.tsv"

    master.to_csv(master_path, sep="\t", index=False)
    dictionary.to_csv(dict_path, sep="\t", index=False)
    with open(qc_path, "w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2, ensure_ascii=False)

    print("Output files:")
    print(f"  {master_path}")
    print(f"  {qc_path}")
    print(f"  {dict_path}")
    print("\nQuality-control summary:")
    print(f"  rows: {qc['n_rows_master']}")
    print(f"  subjects: {qc['n_subjects']}")
    print(f"  condition counts: {qc['condition_counts']}")
    print(f"  rating >100: {qc['rating_raw_over_100_count']}")
    if qc["warnings"]:
        print("\nWarnings:")
        for w in qc["warnings"]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
