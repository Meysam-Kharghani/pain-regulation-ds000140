#!/usr/bin/env python3
"""Diagnose rank deficiency and target-trial estimability in ds000140 LSS designs.

This script reconstructs the design matrices only; it does not load ROI masks or fit
BOLD time series. It reports whether the target_trial coefficient remains uniquely
estimable when nuisance columns are collinear.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd


def load_lss_module(repo_root: Path):
    path = repo_root / "code" / "05_trialwise_connectivity" / "run_trialwise_roi_connectivity.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("lss_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def numeric_run(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def parse_runs(text: str) -> set[int]:
    vals = set()
    for token in text.split(","):
        token = token.strip()
        if token:
            vals.add(int(token))
    return vals


def nonzero_condition_number(s: np.ndarray, tol: float) -> float:
    nz = s[s > tol]
    if len(nz) == 0:
        return math.inf
    return float(nz.max() / nz.min())


def exact_duplicate_groups(X: np.ndarray, names: list[str], atol: float = 1e-10) -> list[list[str]]:
    groups: list[list[str]] = []
    used: set[int] = set()
    for i in range(X.shape[1]):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, X.shape[1]):
            if j in used:
                continue
            if np.allclose(X[:, i], X[:, j], atol=atol, rtol=0.0) or np.allclose(
                X[:, i], -X[:, j], atol=atol, rtol=0.0
            ):
                group.append(j)
                used.add(j)
        if len(group) > 1:
            used.add(i)
            groups.append([names[k] for k in group])
    return groups


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--behavior-trials", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--runs", default="3,7")
    p.add_argument("--bold-column", default="bold_unsmoothed")
    p.add_argument("--rating-events-column", default="events_long")
    p.add_argument("--hrf-model", default="spm")
    p.add_argument("--drift-model", default="cosine")
    p.add_argument("--high-pass", type=float, default=0.008)
    p.add_argument("--other-mode", choices=["all_other", "pooled", "by_condition"], default="all_other")
    p.add_argument("--estimability-tol", type=float, default=1e-8)
    args = p.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    inventory_path = args.inventory.expanduser().resolve()
    behavior_path = args.behavior_trials.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    m = load_lss_module(repo_root)
    inv = pd.read_csv(inventory_path, sep="\t")
    behavior = pd.read_csv(behavior_path, sep="\t") if behavior_path.exists() else None

    inv["run"] = numeric_run(inv["run"])
    inv = inv[inv["run"].isin(parse_runs(args.runs))].copy()
    if "qc_retained" in inv.columns:
        q = inv["qc_retained"]
        if q.dtype == bool:
            inv = inv[q].copy()
        else:
            inv = inv[q.astype(str).str.lower().isin({"true", "1", "yes"})].copy()

    records: list[dict] = []
    failed: list[dict] = []

    for _, row in inv.iterrows():
        try:
            subject, run, run_label = m.get_subject_run(row)
            bold = m.infer_path(
                row,
                ["bold_unsmoothed", "bold", "bold_smoothed", "smoothed_bold", "bold_file", "bold_path", "preproc_bold", "func"],
                "BOLD",
                preferred=args.bold_column,
            )
            events_path = m.infer_path(
                row,
                ["events_stim", "events", "events_file", "events_path", "events_long"],
                "events",
            )
            rating_path = m.infer_optional_path(
                row,
                ["events_long", "events_all", "events_with_ratings", "events"],
                preferred=args.rating_events_column,
            )
            conf_path = None
            for c in ["confounds_glm", "confounds", "confounds_file", "confounds_path"]:
                if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
                    cand = Path(str(row[c])).expanduser()
                    if cand.exists():
                        conf_path = cand.resolve()
                        break

            img = nib.load(str(bold))
            n_vols = int(img.shape[-1])
            tr = m.get_tr(row, img)
            frame_times = np.arange(n_vols, dtype=float) * tr
            events = m.read_events(events_path)
            events = m.maybe_merge_behavior(events, behavior, subject, run)
            rating_events = m.read_rating_events(rating_path)
            rating_source = str(rating_path) if not rating_events.empty else ""
            if rating_events.empty:
                rating_events = m.rating_events_from_trial_table(events)
                if not rating_events.empty:
                    rating_source = "behavior_timing_columns"
            if rating_events.empty:
                raise ValueError("No rating events found")
            confounds = m.read_confounds(conf_path, n_vols)

            for i in range(len(events)):
                lss_events = m.build_lss_events(
                    events,
                    i,
                    other_mode=args.other_mode,
                    rating_events=rating_events,
                )
                try:
                    design = m.make_first_level_design_matrix(
                        frame_times=frame_times,
                        events=lss_events,
                        hrf_model=args.hrf_model,
                        drift_model=args.drift_model,
                        high_pass=args.high_pass,
                        add_regs=confounds if confounds.shape[1] > 0 else None,
                        add_reg_names=list(confounds.columns) if confounds.shape[1] > 0 else None,
                    )
                except TypeError:
                    design = m.make_first_level_design_matrix(
                        frame_times,
                        events=lss_events,
                        hrf_model=args.hrf_model,
                        drift_model=args.drift_model,
                        high_pass=args.high_pass,
                        add_regs=confounds if confounds.shape[1] > 0 else None,
                        add_reg_names=list(confounds.columns) if confounds.shape[1] > 0 else None,
                    )

                names = list(design.columns)
                if "target_trial" not in names:
                    raise ValueError("Design lacks target_trial")
                X = np.nan_to_num(design.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                target_idx = names.index("target_trial")
                rank = int(np.linalg.matrix_rank(X))
                deficit = int(X.shape[1] - rank)

                # Contrast estimability: c must lie in row space of X.
                c = np.zeros(X.shape[1], dtype=float)
                c[target_idx] = 1.0
                row_proj = np.linalg.pinv(X) @ X
                estimability_error = float(np.linalg.norm(c - c @ row_proj))

                # Unique target coefficient: target column must add rank beyond all other columns.
                others = np.delete(X, target_idx, axis=1)
                rank_others = int(np.linalg.matrix_rank(others))
                target_adds_rank = bool(rank > rank_others)
                target_vec = X[:, target_idx]
                target_resid = target_vec - others @ (np.linalg.pinv(others) @ target_vec)
                target_rel_resid = float(np.linalg.norm(target_resid) / max(np.linalg.norm(target_vec), np.finfo(float).eps))

                s = np.linalg.svd(X, compute_uv=False)
                sv_tol = float(max(X.shape) * np.finfo(float).eps * (s[0] if len(s) else 0.0))
                cond_nonzero = nonzero_condition_number(s, sv_tol)
                duplicates = exact_duplicate_groups(X, names)

                ev = events.iloc[i]
                records.append(
                    {
                        "subject": subject,
                        "run": int(run),
                        "run_label": run_label,
                        "trial_lss_index": int(ev["trial_lss_index"]),
                        "condition": str(ev["condition"]),
                        "design_n_rows": int(X.shape[0]),
                        "design_n_cols": int(X.shape[1]),
                        "design_rank": rank,
                        "rank_deficit": deficit,
                        "rank_without_target": rank_others,
                        "target_adds_rank": target_adds_rank,
                        "target_estimability_error": estimability_error,
                        "target_relative_residual": target_rel_resid,
                        "target_estimable": bool(target_adds_rank and estimability_error <= args.estimability_tol),
                        "condition_number_nonzero": cond_nonzero,
                        "smallest_nonzero_singular_value": float(s[s > sv_tol].min()) if np.any(s > sv_tol) else np.nan,
                        "n_exact_duplicate_groups": len(duplicates),
                        "exact_duplicate_groups_json": json.dumps(duplicates, ensure_ascii=False),
                        "n_confounds": int(confounds.shape[1]),
                        "rating_event_source": rating_source,
                        "events_file": str(events_path),
                        "confounds_file": str(conf_path) if conf_path else "",
                    }
                )
        except Exception as exc:
            failed.append({"subject": str(row.get("subject", "")), "run": row.get("run", ""), "error": repr(exc)})

    detail = pd.DataFrame(records)
    failures = pd.DataFrame(failed)
    detail_path = outdir / "lss_design_estimability_per_trial.tsv"
    failure_path = outdir / "lss_design_estimability_failures.tsv"
    detail.to_csv(detail_path, sep="\t", index=False)
    failures.to_csv(failure_path, sep="\t", index=False)

    if detail.empty:
        raise RuntimeError("No designs were diagnosed")

    summary = {
        "n_trials": int(len(detail)),
        "n_runs": int(detail[["subject", "run"]].drop_duplicates().shape[0]),
        "n_subjects": int(detail["subject"].nunique()),
        "n_failures": int(len(failures)),
        "rank_deficit_counts": {str(k): int(v) for k, v in detail["rank_deficit"].value_counts().sort_index().items()},
        "n_target_not_estimable": int((~detail["target_estimable"]).sum()),
        "max_target_estimability_error": float(detail["target_estimability_error"].max()),
        "min_target_relative_residual": float(detail["target_relative_residual"].min()),
        "median_condition_number_nonzero": float(detail["condition_number_nonzero"].median()),
        "max_condition_number_nonzero": float(detail["condition_number_nonzero"].max()),
        "n_trials_with_exact_duplicate_groups": int((detail["n_exact_duplicate_groups"] > 0).sum()),
        "passed": bool(len(failures) == 0 and len(detail) == 660 and detail["target_estimable"].all()),
    }
    (outdir / "lss_design_estimability_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    deficient = detail[detail["rank_deficit"] > 0]
    deficient.to_csv(outdir / "lss_rank_deficient_trials.tsv", sep="\t", index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nPer-trial table: {detail_path}")
    print(f"Summary: {outdir / 'lss_design_estimability_summary.json'}")
    if summary["passed"]:
        print("\nLSS DESIGN ESTIMABILITY CHECK PASSED")
        return 0
    print("\nLSS design-estimability check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
