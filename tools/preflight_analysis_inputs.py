#!/usr/bin/env python3
"""Preflight inputs for the ds000140 imaging analyses.

The check is read-only. It verifies the QC-retained regulation-run inventory,
BOLD/event/confound paths, rating-period rows, and key repository tables before
running the expensive LSS and DCM sensitivity analyses.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def resolve_path(value: object, dataset_root: Path) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("<PROJECT_ROOT>", str(dataset_root))
    text = text.replace("${DS000140_ROOT}", str(dataset_root))
    return Path(os.path.expandvars(text)).expanduser().resolve()


def numeric_run(value: object) -> int | None:
    if isinstance(value, (int, float)) and pd.notna(value):
        return int(value)
    hit = re.search(r"\d+", str(value))
    return int(hit.group()) if hit else None


def find_rating_rows(path: Path) -> tuple[int, int, set[str]]:
    table = pd.read_csv(path, sep="\t")
    phase = table.get("event_phase", pd.Series("", index=table.index)).astype(str).str.lower()
    trial_type = table.get("trial_type", pd.Series("", index=table.index)).astype(str).str.lower()
    rating = phase.eq("rating") | trial_type.str.contains("rating|response", regex=True, na=False)
    stim = phase.eq("stim") | trial_type.str.contains("stim|heat", regex=True, na=False)
    if not stim.any():
        stim = ~rating
    conditions: set[str] = set()
    if "condition" in table.columns:
        values = table.loc[stim, "condition"].dropna().astype(str).str.lower()
        conditions = {x for x in values if x in {"up", "down", "passive"}}
    else:
        values = trial_type[stim]
        for name in ("up", "down", "passive"):
            if values.str.contains(name, na=False).any():
                conditions.add(name)
    return int(stim.sum()), int(rating.sum()), conditions


def first_existing(candidates: Iterable[Path]) -> Path | None:
    return next((p for p in candidates if p.exists()), None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--repo-root", required=True, type=Path)
    ap.add_argument("--inventory", default="")
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    inventory = Path(args.inventory).expanduser().resolve() if args.inventory else first_existing([
        dataset_root / "derivatives/first_level_inputs/tables/first_level_run_inventory.tsv",
        repo_root / "results/glm/first_level_run_inventory.tsv",
    ])
    if inventory is None or not inventory.exists():
        raise FileNotFoundError("Could not locate first_level_run_inventory.tsv")

    inv = pd.read_csv(inventory, sep="\t")
    if "status" in inv.columns:
        status = inv["status"].astype(str).str.upper().str.strip()
        if status.eq("READY").any():
            inv = inv[status.eq("READY")].copy()
        elif status.eq("OK").any():
            inv = inv[status.eq("OK")].copy()
        else:
            inv = inv[~status.str.contains("FAIL", na=False)].copy()
    inv["_run"] = inv["run"].map(numeric_run)
    reg = inv[inv["_run"].isin([3, 7])].copy()

    rows: list[dict[str, object]] = []
    for _, row in reg.iterrows():
        subject = str(row.get("subject", ""))
        run = int(row["_run"])
        bold = resolve_path(row.get("bold_unsmoothed", row.get("bold", "")), dataset_root)
        events = resolve_path(row.get("events_long", row.get("events", "")), dataset_root)
        confounds = resolve_path(row.get("confounds_glm", row.get("confounds", "")), dataset_root)
        stim_n = rating_n = 0
        conditions: set[str] = set()
        event_error = ""
        if events and events.exists():
            try:
                stim_n, rating_n, conditions = find_rating_rows(events)
            except Exception as exc:  # pragma: no cover - external file dependent
                event_error = f"{type(exc).__name__}: {exc}"
        rows.append({
            "subject": subject,
            "run": run,
            "bold_path": str(bold or ""),
            "bold_exists": bool(bold and bold.exists()),
            "events_path": str(events or ""),
            "events_exists": bool(events and events.exists()),
            "confounds_path": str(confounds or ""),
            "confounds_exists": bool(confounds and confounds.exists()),
            "n_stimulation_events": stim_n,
            "n_rating_events": rating_n,
            "conditions": ",".join(sorted(conditions)),
            "event_error": event_error,
        })

    report = pd.DataFrame(rows).sort_values(["subject", "run"])
    report.to_csv(outdir / "analysis_input_preflight_runs.tsv", sep="\t", index=False)

    expected_files = {
        "roi_definitions": first_existing([
            dataset_root / "derivatives/roi_masks/roi_definitions.tsv",
            repo_root / "results/roi_signatures/roi_definitions.tsv",
        ]),
        "behavior_trials": first_existing([
            dataset_root / "derivatives/beh/behavior_trials_master_clean.tsv",
            repo_root / "results/behavior/behavior_trials_master_clean.tsv.gz",
        ]),
        "screening_matrix": repo_root / "results/trialwise_connectivity/screening_predictor_outcome_matrix.tsv",
        "reported_controls": repo_root / "results/sensitivity_audits/matched_coupling_controls",
    }

    summary = {
        "inventory": str(inventory),
        "n_qc_retained_all_runs": int(len(inv)),
        "n_regulation_runs": int(len(reg)),
        "n_subjects_regulation": int(reg["subject"].nunique()) if "subject" in reg else None,
        "run_counts": {str(k): int(v) for k, v in reg["_run"].value_counts().sort_index().items()},
        "all_bold_exist": bool(report["bold_exists"].all()) if not report.empty else False,
        "all_events_exist": bool(report["events_exists"].all()) if not report.empty else False,
        "all_confounds_exist": bool(report["confounds_exists"].all()) if not report.empty else False,
        "rating_event_counts": {str(k): int(v) for k, v in report["n_rating_events"].value_counts().sort_index().items()},
        "stimulation_event_counts": {str(k): int(v) for k, v in report["n_stimulation_events"].value_counts().sort_index().items()},
        "required_resources": {k: {"path": str(v or ""), "exists": bool(v and v.exists())} for k, v in expected_files.items()},
    }
    problems: list[str] = []
    if len(inv) != 288:
        problems.append(f"Expected 288 QC-retained runs, found {len(inv)}.")
    if len(reg) != 66:
        problems.append(f"Expected 66 regulation runs, found {len(reg)}.")
    if "subject" in reg and reg["subject"].nunique() != 33:
        problems.append(f"Expected 33 regulation participants, found {reg['subject'].nunique()}.")
    for field in ("bold_exists", "events_exists", "confounds_exists"):
        if not report.empty and not report[field].all():
            problems.append(f"One or more regulation runs have {field}=False.")
    if not report.empty and not report["n_rating_events"].eq(10).all():
        problems.append("Not every regulation run has exactly 10 rating events.")
    if not report.empty and not report["n_stimulation_events"].eq(10).all():
        problems.append("Not every regulation run has exactly 10 stimulation events.")
    for name, info in summary["required_resources"].items():
        if not info["exists"]:
            problems.append(f"Required resource missing: {name}")
    summary["problems"] = problems
    summary["passed"] = not problems
    (outdir / "analysis_input_preflight_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if problems:
        print("\nPREFLIGHT FAILED")
        return 1
    print("\nPREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
