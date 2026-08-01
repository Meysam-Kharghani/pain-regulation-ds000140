#!/usr/bin/env python3
"""Estimate rating-adjusted trialwise Neurologic Pain Signature expression.

The estimator uses the same LSS event construction, confounds, rating-period
nuisance regressor, and Frisch-Waugh-Lovell target residualization as the
trialwise ROI workflow. The target-trial beta image is projected onto the NPS
weights without writing intermediate beta images.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from scipy import stats

CONNECTIVITY_DIR = Path(__file__).resolve().parents[1] / "05_trialwise_connectivity"
sys.path.insert(0, str(CONNECTIVITY_DIR))
import run_trialwise_roi_connectivity as core  # noqa: E402


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def load_signature_timeseries(
    bold_path: Path,
    signature_path: Path,
) -> tuple[np.ndarray, np.ndarray, int]:
    bold_img = nib.load(str(bold_path))
    signature_img = nib.load(str(signature_path))
    signature_resampled = resample_to_img(
        signature_img,
        bold_img,
        interpolation="continuous",
        force_resample=True,
        copy_header=True,
    )
    weights_3d = np.asarray(signature_resampled.get_fdata(), dtype=np.float64)
    mask = np.isfinite(weights_3d) & (np.abs(weights_3d) > 1e-12)
    if not np.any(mask):
        raise ValueError("The resampled signature contains no nonzero finite weights.")
    weights = weights_3d[mask]
    bold_4d = np.asarray(bold_img.dataobj, dtype=np.float64)
    signals = bold_4d[mask, :].T
    if signals.shape[0] != bold_img.shape[-1]:
        raise ValueError("Signature time-series length does not match the BOLD run.")
    finite_voxels = np.isfinite(signals).all(axis=0) & np.isfinite(weights)
    signals = signals[:, finite_voxels]
    weights = weights[finite_voxels]
    if signals.shape[1] == 0:
        raise ValueError("No finite signature voxels remain after BOLD extraction.")
    return signals, weights, int(signals.shape[1])


def target_residual(
    design: pd.DataFrame,
    solver_rcond: float,
    target_unique_tol: float,
) -> tuple[np.ndarray, float, float]:
    if "target_trial" not in design.columns:
        raise ValueError("LSS design lacks the target_trial column.")
    matrix = np.nan_to_num(
        np.asarray(design, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    target_index = list(design.columns).index("target_trial")
    target = matrix[:, target_index]
    nuisance = np.delete(matrix, target_index, axis=1)
    norms = np.linalg.norm(nuisance, axis=0)
    nuisance = nuisance[:, norms > np.finfo(float).eps]
    norms = norms[norms > np.finfo(float).eps]
    if nuisance.shape[1]:
        nuisance_scaled = nuisance / norms
        residual = target - nuisance_scaled @ (
            np.linalg.pinv(nuisance_scaled, rcond=solver_rcond) @ target
        )
    else:
        residual = target.copy()
    target_norm = max(float(np.linalg.norm(target)), np.finfo(float).eps)
    relative_residual = float(np.linalg.norm(residual) / target_norm)
    if relative_residual <= target_unique_tol:
        raise ValueError(
            "Target-trial coefficient is not uniquely estimable: "
            f"relative residual={relative_residual:.6g}."
        )
    denominator = float(residual @ target)
    if not np.isfinite(denominator) or abs(denominator) <= np.finfo(float).eps:
        raise ValueError("Invalid target-trial FWL denominator.")
    return residual, denominator, relative_residual


def fit_run(
    inventory_row: pd.Series,
    signature_path: Path,
    behavior: pd.DataFrame | None,
    args: argparse.Namespace,
) -> pd.DataFrame:
    subject, run, run_label = core.get_subject_run(inventory_row)
    bold_path = core.infer_path(
        inventory_row,
        [
            "bold_unsmoothed",
            "bold",
            "preproc_bold",
            "bold_file",
            "bold_path",
        ],
        "BOLD",
        preferred=args.bold_column,
    )
    events_path = core.infer_path(
        inventory_row,
        ["events_stim", "events", "events_file", "events_path", "events_long"],
        "events",
    )
    rating_path = core.infer_optional_path(
        inventory_row,
        ["events_long", "events_all", "events_with_ratings", "events"],
        preferred=args.rating_events_column,
    )
    confounds_path = core.infer_optional_path(
        inventory_row,
        ["confounds_glm", "confounds", "confounds_file", "confounds_path"],
    )

    bold_img = nib.load(str(bold_path))
    n_volumes = int(bold_img.shape[-1])
    repetition_time = core.get_tr(inventory_row, bold_img)
    frame_times = np.arange(n_volumes, dtype=float) * repetition_time

    events = core.read_events(events_path)
    events = core.maybe_merge_behavior(events, behavior, subject, run)
    rating_events = core.read_rating_events(rating_path)
    if rating_events.empty:
        rating_events = core.rating_events_from_trial_table(events)
    if rating_events.empty:
        raise ValueError("Rating periods could not be recovered for this run.")

    confounds = core.read_confounds(confounds_path, n_volumes)
    signals, weights, n_voxels = load_signature_timeseries(
        bold_path,
        signature_path,
    )
    absolute_weight_sum = float(np.sum(np.abs(weights)))
    if absolute_weight_sum <= 0:
        raise ValueError("Signature absolute-weight norm is zero.")

    rows: list[dict[str, object]] = []
    for trial_index in range(len(events)):
        lss_events = core.build_lss_events(
            events,
            trial_index,
            other_mode=args.other_mode,
            rating_events=rating_events,
        )
        try:
            design = core.make_first_level_design_matrix(
                frame_times=frame_times,
                events=lss_events,
                hrf_model=args.hrf_model,
                drift_model=args.drift_model,
                high_pass=args.high_pass,
                add_regs=confounds if confounds.shape[1] else None,
                add_reg_names=list(confounds.columns) if confounds.shape[1] else None,
            )
        except TypeError:
            design = core.make_first_level_design_matrix(
                frame_times,
                events=lss_events,
                hrf_model=args.hrf_model,
                drift_model=args.drift_model,
                high_pass=args.high_pass,
                add_regs=confounds if confounds.shape[1] else None,
                add_reg_names=list(confounds.columns) if confounds.shape[1] else None,
            )

        residual, denominator, relative_residual = target_residual(
            design,
            args.solver_rcond,
            args.target_unique_tol,
        )
        beta_voxels = (residual @ signals) / denominator
        signature_dot = float(beta_voxels @ weights)
        event = events.iloc[trial_index]
        rows.append(
            {
                "subject": subject,
                "run": run,
                "run_label": run_label,
                "trial_lss_index": int(event["trial_lss_index"]),
                "condition": event["condition"],
                "onset": float(event["onset"]),
                "duration": float(event["duration"]),
                "temperature_for_model": event.get("temperature_for_model", np.nan),
                "rating_for_model": event.get("rating_for_model", np.nan),
                "nps_dot": signature_dot,
                "nps_weighted_mean": signature_dot / absolute_weight_sum,
                "n_signature_voxels": n_voxels,
                "target_relative_residual": relative_residual,
                "solver_rcond": args.solver_rcond,
                "target_unique_tol": args.target_unique_tol,
                "rating_nuisance_included": True,
            }
        )
    return pd.DataFrame(rows)


def one_sample(values: pd.Series, label: str) -> dict[str, object]:
    sample = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(sample) < 2:
        return {"contrast": label, "n": len(sample)}
    test = stats.ttest_1samp(sample, 0.0)
    return {
        "contrast": label,
        "n": len(sample),
        "mean": float(sample.mean()),
        "sd": float(sample.std(ddof=1)),
        "t": float(test.statistic),
        "p": float(test.pvalue),
    }


def summarize(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_condition = (
        trials.groupby(["subject", "condition"], as_index=False)
        .agg(
            n_trials=("nps_weighted_mean", "size"),
            nps_weighted_mean=("nps_weighted_mean", "mean"),
        )
    )
    wide = subject_condition.pivot(
        index="subject",
        columns="condition",
        values="nps_weighted_mean",
    )
    rows: list[dict[str, object]] = []
    definitions = {
        "up_minus_passive": ("up", "passive"),
        "down_minus_passive": ("down", "passive"),
        "up_minus_down": ("up", "down"),
    }
    for label, (left, right) in definitions.items():
        if left in wide.columns and right in wide.columns:
            rows.append(one_sample(wide[left] - wide[right], label))

    passive_slopes = []
    for subject, frame in trials[trials["condition"] == "passive"].groupby("subject"):
        valid = frame[["temperature_for_model", "nps_weighted_mean"]].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()
        if len(valid) >= 3 and valid["temperature_for_model"].nunique() > 1:
            slope = stats.linregress(
                valid["temperature_for_model"],
                valid["nps_weighted_mean"],
            ).slope
            passive_slopes.append({"subject": subject, "passive_temperature_slope": slope})
    slope_frame = pd.DataFrame(passive_slopes)
    if not slope_frame.empty:
        rows.append(
            one_sample(
                slope_frame["passive_temperature_slope"],
                "passive_temperature_slope",
            )
        )
    return subject_condition, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--nps-map", type=Path, required=True)
    parser.add_argument("--behavior-trials", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bold-column", default="bold_unsmoothed")
    parser.add_argument("--rating-events-column", default="events_long")
    parser.add_argument("--hrf-model", default="spm")
    parser.add_argument("--drift-model", default="cosine")
    parser.add_argument("--high-pass", type=float, default=0.008)
    parser.add_argument("--other-mode", choices=["all_other", "by_condition"], default="all_other")
    parser.add_argument("--solver-rcond", type=float, default=1e-12)
    parser.add_argument("--target-unique-tol", type=float, default=1e-6)
    parser.add_argument("--runs", default="")
    parser.add_argument("--max-runs", type=int, default=0)
    args = parser.parse_args()

    args.inventory = args.inventory.expanduser().resolve()
    args.nps_map = args.nps_map.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()
    if not args.inventory.exists():
        raise FileNotFoundError(args.inventory)
    if not args.nps_map.exists():
        raise FileNotFoundError(args.nps_map)
    args.outdir.mkdir(parents=True, exist_ok=True)

    inventory = read_table(args.inventory)
    if "status" in inventory.columns:
        status = inventory["status"].astype(str).str.upper()
        if (status == "READY").any():
            inventory = inventory[status == "READY"].copy()
        elif (status == "OK").any():
            inventory = inventory[status == "OK"].copy()
        else:
            inventory = inventory[~status.str.contains("FAIL", na=False)].copy()
    if args.runs:
        selected_runs = {int(value.strip()) for value in args.runs.split(",") if value.strip()}
        inventory = inventory[
            pd.to_numeric(inventory["run"], errors="coerce").isin(selected_runs)
        ].copy()
    if args.max_runs > 0:
        inventory = inventory.head(args.max_runs).copy()

    behavior = None
    if args.behavior_trials:
        behavior_path = args.behavior_trials.expanduser().resolve()
        if not behavior_path.exists():
            raise FileNotFoundError(behavior_path)
        behavior = read_table(behavior_path)

    run_tables = []
    run_log = []
    for _, inventory_row in inventory.iterrows():
        subject = str(inventory_row.get("subject", "unknown"))
        run = inventory_row.get("run", np.nan)
        try:
            frame = fit_run(inventory_row, args.nps_map, behavior, args)
            run_tables.append(frame)
            run_log.append(
                {
                    "subject": frame["subject"].iloc[0],
                    "run": frame["run"].iloc[0],
                    "status": "OK",
                    "n_trials": len(frame),
                }
            )
        except Exception as error:
            run_log.append(
                {
                    "subject": subject,
                    "run": run,
                    "status": "FAILED",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            )

    run_inventory = pd.DataFrame(run_log)
    write_table(run_inventory, args.outdir / "run_inventory.tsv")
    if not run_tables:
        raise RuntimeError("No run completed successfully.")
    trials = pd.concat(run_tables, ignore_index=True)
    subject_condition, tests = summarize(trials)
    write_table(trials, args.outdir / "trialwise_nps_fwl_lss.tsv.gz")
    write_table(subject_condition, args.outdir / "participant_condition_nps.tsv")
    write_table(tests, args.outdir / "nps_condition_tests.tsv")
    (args.outdir / "analysis_config.json").write_text(
        json.dumps(
            {
                "inventory": str(args.inventory),
                "nps_map": str(args.nps_map),
                "rating_nuisance_included": True,
                "solver": "Frisch-Waugh-Lovell target residualization",
                "solver_rcond": args.solver_rcond,
                "target_unique_tol": args.target_unique_tol,
                "n_runs_ok": int((run_inventory["status"] == "OK").sum()),
                "n_trials": int(len(trials)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
