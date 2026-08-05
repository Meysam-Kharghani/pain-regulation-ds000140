#!/usr/bin/env python3
"""AR(1)-prewhitened sensitivity analyses for trialwise ROI and NPS estimates.

This script reuses the repository's event construction, confound handling,
ROI extraction, and LSS design specification, but estimates each LSS model
with Nilearn's ``run_glm(..., noise_model="ar1")`` instead of ordinary least
squares. It provides two analysis modes:

``roi``
    Re-estimate trialwise ROI beta series with AR(1) temporal noise modeling.
    The resulting wide table can be passed directly to
    ``run_matched_coupling_controls.py``.

``nps``
    Re-estimate the rating-adjusted trialwise NPS expression after projecting
    each run onto the supplied NPS weights. This is an AR(1) sensitivity at the
    signature-expression time-series level; the NPS weight map is read from a
    user-supplied path and is never copied to the output.

The script imports the canonical event, confound, ROI, and NPS helpers
from the surrounding repository checkout.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable

import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
from nilearn.glm.first_level import run_glm

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
CONNECTIVITY_DIR = REPOSITORY_ROOT / "code" / "05_trialwise_connectivity"
NPS_DIR = REPOSITORY_ROOT / "code" / "04_roi_signatures"
if not CONNECTIVITY_DIR.exists():
    raise RuntimeError(
        "Repository layout was not found. Place this script in "
        "code/07_sensitivity_audits before running it."
    )
sys.path.insert(0, str(CONNECTIVITY_DIR))
sys.path.insert(0, str(NPS_DIR))

import run_trialwise_roi_connectivity as core  # noqa: E402
import estimate_trialwise_nps as nps_core  # noqa: E402


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def portable_path(path: Path) -> str:
    """Return a repository- or dataset-relative path for public metadata."""
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        pass
    parts = resolved.parts
    if "derivatives" in parts:
        index = parts.index("derivatives")
        return Path(*parts[index:]).as_posix()
    return resolved.name


def select_inventory(inventory: pd.DataFrame, runs: str, max_runs: int) -> pd.DataFrame:
    selected = inventory.copy()
    if "status" in selected.columns:
        status = selected["status"].astype(str).str.upper()
        if status.eq("READY").any():
            selected = selected.loc[status.eq("READY")].copy()
        elif status.eq("OK").any():
            selected = selected.loc[status.eq("OK")].copy()
        else:
            selected = selected.loc[~status.str.contains("FAIL", na=False)].copy()
    if runs:
        wanted = {int(value.strip()) for value in runs.split(",") if value.strip()}
        selected = selected.loc[
            pd.to_numeric(selected["run"], errors="coerce").isin(wanted)
        ].copy()
    if max_runs > 0:
        selected = selected.head(max_runs).copy()
    return selected.reset_index(drop=True)


def make_design(
    frame_times: np.ndarray,
    lss_events: pd.DataFrame,
    confounds: pd.DataFrame,
    hrf_model: str,
    drift_model: str,
    high_pass: float,
) -> pd.DataFrame:
    kwargs = {
        "frame_times": frame_times,
        "events": lss_events,
        "hrf_model": hrf_model,
        "drift_model": drift_model,
        "high_pass": high_pass,
        "add_regs": confounds if confounds.shape[1] else None,
        "add_reg_names": list(confounds.columns) if confounds.shape[1] else None,
    }
    try:
        design = core.make_first_level_design_matrix(**kwargs)
    except TypeError:
        design = core.make_first_level_design_matrix(
            frame_times,
            events=lss_events,
            hrf_model=hrf_model,
            drift_model=drift_model,
            high_pass=high_pass,
            add_regs=kwargs["add_regs"],
            add_reg_names=kwargs["add_reg_names"],
        )
    if "target_trial" not in design.columns:
        raise ValueError("LSS design lacks target_trial.")
    if not np.isfinite(np.asarray(design, dtype=float)).all():
        raise ValueError("LSS design contains non-finite values.")
    return design


def target_beta_from_ar1(
    response: np.ndarray,
    design: pd.DataFrame,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an AR(1) GLM and return target coefficients and AR labels."""
    response = np.asarray(response, dtype=np.float64)
    if response.ndim == 1:
        response = response[:, None]
    matrix = np.asarray(design, dtype=np.float64)
    if response.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"Response/design length mismatch: {response.shape[0]} != {matrix.shape[0]}"
        )
    target_index = list(design.columns).index("target_trial")
    labels, results = run_glm(
        response,
        matrix,
        noise_model="ar1",
        bins=bins,
        n_jobs=1,
        verbose=0,
    )
    labels = np.asarray(labels)
    beta = np.full(response.shape[1], np.nan, dtype=float)
    for label, result in results.items():
        indices = np.flatnonzero(labels == label)
        theta = np.asarray(result.theta, dtype=float)
        if theta.ndim == 1:
            theta = theta[:, None]
        values = np.asarray(theta[target_index, :], dtype=float).reshape(-1)
        if values.size != indices.size:
            raise RuntimeError(
                "Nilearn AR(1) result dimensions did not match the assigned series."
            )
        beta[indices] = values
    if not np.isfinite(beta).all():
        raise RuntimeError("AR(1) target coefficients contain non-finite values.")
    return beta, labels


def numeric_ar_labels(labels: Iterable[object]) -> np.ndarray:
    values = pd.to_numeric(pd.Series(list(labels)), errors="coerce").to_numpy(float)
    return values[np.isfinite(values)]


def resolve_run_inputs(
    row: pd.Series,
    behavior: pd.DataFrame | None,
    bold_column: str,
    rating_events_column: str,
) -> dict[str, object]:
    subject, run, run_label = core.get_subject_run(row)
    bold_path = core.infer_path(
        row,
        [
            "bold_unsmoothed",
            "bold",
            "bold_smoothed",
            "smoothed_bold",
            "bold_file",
            "bold_path",
            "preproc_bold",
            "func",
        ],
        "BOLD",
        preferred=bold_column,
    )
    events_path = core.infer_path(
        row,
        ["events_stim", "events", "events_file", "events_path", "events_long"],
        "events",
    )
    rating_path = core.infer_optional_path(
        row,
        ["events_long", "events_all", "events_with_ratings", "events"],
        preferred=rating_events_column,
    )
    confounds_path = core.infer_optional_path(
        row,
        ["confounds_glm", "confounds", "confounds_file", "confounds_path"],
    )
    image = nib.load(str(bold_path))
    n_volumes = int(image.shape[-1])
    repetition_time = core.get_tr(row, image)
    frame_times = np.arange(n_volumes, dtype=float) * repetition_time
    events = core.read_events(events_path)
    events = core.maybe_merge_behavior(events, behavior, subject, run)
    rating_events = core.read_rating_events(rating_path)
    if rating_events.empty:
        rating_events = core.rating_events_from_trial_table(events)
    if rating_events.empty:
        raise ValueError("Rating periods could not be recovered for this run.")
    confounds = core.read_confounds(confounds_path, n_volumes)
    return {
        "subject": subject,
        "run": run,
        "run_label": run_label,
        "bold_path": bold_path,
        "events_path": events_path,
        "rating_path": rating_path,
        "confounds_path": confounds_path,
        "n_volumes": n_volumes,
        "repetition_time": repetition_time,
        "frame_times": frame_times,
        "events": events,
        "rating_events": rating_events,
        "confounds": confounds,
    }


def fit_roi_run(
    inventory_row: pd.Series,
    roi_definitions: pd.DataFrame,
    behavior: pd.DataFrame | None,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = resolve_run_inputs(
        inventory_row,
        behavior,
        args.bold_column,
        args.rating_events_column,
    )
    response, roi_names, _, roi_meta = core.extract_roi_timeseries(
        bold_path=inputs["bold_path"],
        roi_defs=roi_definitions,
        cache_dir=args.outdir / "mask_cache" / inputs["subject"] / f"run-{inputs['run']:02d}",
    )
    rows: list[dict[str, object]] = []
    rho_rows: list[dict[str, object]] = []
    events = inputs["events"]
    for trial_index in range(len(events)):
        lss_events = core.build_lss_events(
            events,
            trial_index,
            other_mode=args.other_mode,
            rating_events=inputs["rating_events"],
        )
        design = make_design(
            inputs["frame_times"],
            lss_events,
            inputs["confounds"],
            args.hrf_model,
            args.drift_model,
            args.high_pass,
        )
        beta, labels = target_beta_from_ar1(response, design, args.ar1_bins)
        event = events.iloc[trial_index]
        base: dict[str, object] = {
            "subject": inputs["subject"],
            "run": inputs["run"],
            "run_label": inputs["run_label"],
            "trial_lss_index": int(event["trial_lss_index"]),
            "condition": event["condition"],
            "onset": float(event["onset"]),
            "duration": float(event["duration"]),
            "temperature_for_model": event.get("temperature_for_model", np.nan),
            "rating_for_model": event.get("rating_for_model", np.nan),
            "n_vols": inputs["n_volumes"],
            "tr": inputs["repetition_time"],
            "design_n_cols": int(design.shape[1]),
            "design_rank": int(np.linalg.matrix_rank(np.asarray(design, dtype=float))),
            "temporal_noise_model": "nilearn_ar1",
            "ar1_bins": args.ar1_bins,
            "rating_nuisance_included": True,
            "bold": portable_path(inputs["bold_path"]),
            "events_file": portable_path(inputs["events_path"]),
            "confounds_file": str(inputs["confounds_path"] or ""),
        }
        for name, value in zip(roi_names, beta):
            base[f"roi_{name}"] = float(value)
        rows.append(base)
        rho_values = numeric_ar_labels(labels)
        rho_rows.append(
            {
                "subject": inputs["subject"],
                "run": inputs["run"],
                "trial_lss_index": int(event["trial_lss_index"]),
                "n_roi_series": len(roi_names),
                "ar1_rho_min": float(np.min(rho_values)) if rho_values.size else np.nan,
                "ar1_rho_median": float(np.median(rho_values)) if rho_values.size else np.nan,
                "ar1_rho_max": float(np.max(rho_values)) if rho_values.size else np.nan,
            }
        )
    rho_frame = pd.DataFrame(rho_rows)
    rho_frame["n_rois_extracted"] = int(roi_meta["n_rois_extracted"])
    return pd.DataFrame(rows), rho_frame


def fit_nps_run(
    inventory_row: pd.Series,
    behavior: pd.DataFrame | None,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = resolve_run_inputs(
        inventory_row,
        behavior,
        args.bold_column,
        args.rating_events_column,
    )
    signals, weights, n_voxels = nps_core.load_signature_timeseries(
        inputs["bold_path"],
        args.nps_map,
    )
    absolute_weight_sum = float(np.sum(np.abs(weights)))
    if absolute_weight_sum <= 0:
        raise ValueError("NPS absolute-weight norm is zero.")
    projected = (signals @ weights) / absolute_weight_sum
    rows: list[dict[str, object]] = []
    rho_rows: list[dict[str, object]] = []
    events = inputs["events"]
    for trial_index in range(len(events)):
        lss_events = core.build_lss_events(
            events,
            trial_index,
            other_mode=args.other_mode,
            rating_events=inputs["rating_events"],
        )
        design = make_design(
            inputs["frame_times"],
            lss_events,
            inputs["confounds"],
            args.hrf_model,
            args.drift_model,
            args.high_pass,
        )
        beta, labels = target_beta_from_ar1(projected[:, None], design, args.ar1_bins)
        weighted_mean = float(beta[0])
        event = events.iloc[trial_index]
        rho_values = numeric_ar_labels(labels)
        rho = float(rho_values[0]) if rho_values.size else np.nan
        rows.append(
            {
                "subject": inputs["subject"],
                "run": inputs["run"],
                "run_label": inputs["run_label"],
                "trial_lss_index": int(event["trial_lss_index"]),
                "condition": event["condition"],
                "onset": float(event["onset"]),
                "duration": float(event["duration"]),
                "temperature_for_model": event.get("temperature_for_model", np.nan),
                "rating_for_model": event.get("rating_for_model", np.nan),
                "nps_dot": weighted_mean * absolute_weight_sum,
                "nps_weighted_mean": weighted_mean,
                "n_signature_voxels": n_voxels,
                "temporal_noise_model": "nilearn_ar1_on_projected_nps_timeseries",
                "ar1_rho": rho,
                "ar1_bins": args.ar1_bins,
                "rating_nuisance_included": True,
                "bold": portable_path(inputs["bold_path"]),
            }
        )
        rho_rows.append(
            {
                "subject": inputs["subject"],
                "run": inputs["run"],
                "trial_lss_index": int(event["trial_lss_index"]),
                "ar1_rho": rho,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(rho_rows)


def load_behavior(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return read_table(resolved)


def run_roi(args: argparse.Namespace) -> None:
    roi_definitions = core.load_roi_definitions(
        args.roi_definitions,
        roi_regex=args.roi_regex,
        exclude_regex=args.exclude_roi_regex,
    )
    inventory = select_inventory(read_table(args.inventory), args.runs, args.max_runs)
    behavior = load_behavior(args.behavior_trials)
    trial_frames: list[pd.DataFrame] = []
    rho_frames: list[pd.DataFrame] = []
    run_log: list[dict[str, object]] = []
    per_run = args.outdir / "per_run"
    per_run.mkdir(parents=True, exist_ok=True)
    for _, row in inventory.iterrows():
        subject = str(row.get("subject", "unknown"))
        run = row.get("run", np.nan)
        try:
            trials, rhos = fit_roi_run(row, roi_definitions, behavior, args)
            trial_frames.append(trials)
            rho_frames.append(rhos)
            write_table(
                trials,
                per_run / f"{trials['subject'].iloc[0]}_run-{int(trials['run'].iloc[0]):02d}_roi_ar1_lss.tsv.gz",
            )
            run_log.append(
                {
                    "subject": trials["subject"].iloc[0],
                    "run": int(trials["run"].iloc[0]),
                    "status": "OK",
                    "n_trials": len(trials),
                    "n_rois": len([column for column in trials.columns if column.startswith("roi_")]),
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
    write_table(pd.DataFrame(run_log), args.outdir / "tables" / "run_inventory_roi_ar1.tsv")
    if not trial_frames:
        raise RuntimeError("No ROI run completed successfully.")
    trials = pd.concat(trial_frames, ignore_index=True)
    rhos = pd.concat(rho_frames, ignore_index=True)
    write_table(trials, args.outdir / "tables" / "trialwise_roi_lss_wide.tsv.gz")
    write_table(rhos, args.outdir / "tables" / "ar1_rho_by_trial_and_run.tsv.gz")


def run_nps(args: argparse.Namespace) -> None:
    inventory = select_inventory(read_table(args.inventory), args.runs, args.max_runs)
    behavior = load_behavior(args.behavior_trials)
    trial_frames: list[pd.DataFrame] = []
    rho_frames: list[pd.DataFrame] = []
    run_log: list[dict[str, object]] = []
    for _, row in inventory.iterrows():
        subject = str(row.get("subject", "unknown"))
        run = row.get("run", np.nan)
        try:
            trials, rhos = fit_nps_run(row, behavior, args)
            trial_frames.append(trials)
            rho_frames.append(rhos)
            run_log.append(
                {
                    "subject": trials["subject"].iloc[0],
                    "run": int(trials["run"].iloc[0]),
                    "status": "OK",
                    "n_trials": len(trials),
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
    write_table(pd.DataFrame(run_log), args.outdir / "run_inventory_nps_ar1.tsv")
    if not trial_frames:
        raise RuntimeError("No NPS run completed successfully.")
    trials = pd.concat(trial_frames, ignore_index=True)
    rhos = pd.concat(rho_frames, ignore_index=True)
    subject_condition, tests = nps_core.summarize(trials)
    write_table(trials, args.outdir / "trialwise_nps_ar1_lss.tsv.gz")
    write_table(rhos, args.outdir / "ar1_rho_by_trial_and_run.tsv.gz")
    write_table(subject_condition, args.outdir / "participant_condition_nps.tsv")
    write_table(tests, args.outdir / "nps_condition_tests.tsv")


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--behavior-trials", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bold-column", default="bold_unsmoothed")
    parser.add_argument("--rating-events-column", default="events_long")
    parser.add_argument("--hrf-model", default="spm")
    parser.add_argument("--drift-model", default="cosine")
    parser.add_argument("--high-pass", type=float, default=0.008)
    parser.add_argument("--other-mode", choices=["all_other", "by_condition"], default="all_other")
    parser.add_argument("--ar1-bins", type=int, default=100)
    parser.add_argument("--runs", default="")
    parser.add_argument("--max-runs", type=int, default=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    roi_parser = subparsers.add_parser("roi", help="AR(1)-prewhitened ROI LSS estimates")
    add_shared_arguments(roi_parser)
    roi_parser.add_argument("--roi-definitions", type=Path, required=True)
    roi_parser.add_argument("--roi-regex", default="")
    roi_parser.add_argument("--exclude-roi-regex", default="")

    nps_parser = subparsers.add_parser("nps", help="AR(1)-prewhitened NPS LSS estimates")
    add_shared_arguments(nps_parser)
    nps_parser.add_argument("--nps-map", type=Path, required=True)

    args = parser.parse_args()
    args.inventory = args.inventory.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.inventory.exists():
        raise FileNotFoundError(args.inventory)
    if hasattr(args, "roi_definitions"):
        args.roi_definitions = args.roi_definitions.expanduser().resolve()
        if not args.roi_definitions.exists():
            raise FileNotFoundError(args.roi_definitions)
        run_roi(args)
    else:
        args.nps_map = args.nps_map.expanduser().resolve()
        if not args.nps_map.exists():
            raise FileNotFoundError(args.nps_map)
        run_nps(args)

    config = {
        "completed_at": now_iso(),
        "mode": "roi" if hasattr(args, "roi_definitions") else "nps",
        "inventory": portable_path(args.inventory),
        "outdir": portable_path(args.outdir),
        "temporal_noise_model": "nilearn run_glm AR(1)",
        "ar1_bins": args.ar1_bins,
        "nilearn": nilearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "python": platform.python_version(),
    }
    if hasattr(args, "nps_map"):
        config["nps_map"] = portable_path(args.nps_map)
        config["nps_note"] = (
            "AR(1) was fitted to the NPS-weighted run time series; the supplied "
            "weight map was read in place and was not copied."
        )
    (args.outdir / "analysis_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print(f"AR(1) sensitivity outputs written to {args.outdir}")


if __name__ == "__main__":
    main()
