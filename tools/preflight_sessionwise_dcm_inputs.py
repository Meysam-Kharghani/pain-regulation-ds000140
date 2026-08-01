#!/usr/bin/env python3
"""Preflight the all-passive, run-separated DCM analysis for ds000140."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    return p.parse_args()


def resolve_path(value: Any, root: Path) -> Path:
    text = str(value).strip()
    text = text.replace("<PROJECT_ROOT>", str(root))
    text = text.replace("${DS000140_ROOT}", str(root))
    return Path(text).expanduser()


def normalize_subject(value: Any) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return f"sub-{int(digits):02d}"


def keep_inventory(df: pd.DataFrame) -> pd.DataFrame:
    keep = pd.Series(True, index=df.index)
    if "status" in df:
        status = df["status"].astype(str).str.strip().str.upper()
        if status.eq("READY").any():
            keep &= status.eq("READY")
        elif status.eq("OK").any():
            keep &= status.eq("OK")
        else:
            keep &= ~status.str.contains("FAIL", na=False)
    if "analysis_decision" in df:
        keep &= (
            df["analysis_decision"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.startswith("KEEP")
        )
    elif "keep_main" in df:
        keep &= (
            df["keep_main"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "y", "t"])
        )
    out = df.loc[keep].copy()
    out["subject"] = out["subject"].map(normalize_subject)
    out["run"] = pd.to_numeric(out["run"], errors="raise").astype(int)
    return out.sort_values(["subject", "run"]).reset_index(drop=True)


def infer_condition(events_path: Path) -> tuple[str, int, int]:
    events = pd.read_csv(events_path, sep="\t")
    if "event_phase" in events:
        phase = events["event_phase"].astype(str).str.lower()
    else:
        phase = pd.Series("", index=events.index)
    if "trial_type" in events:
        trial_type = events["trial_type"].astype(str).str.lower()
    else:
        trial_type = pd.Series("", index=events.index)

    is_rating = phase.eq("rating") | trial_type.str.contains("rating|response", regex=True)
    is_stim = phase.eq("stim") | trial_type.str.contains("stim|heat", regex=True)
    if not is_stim.any():
        is_stim = ~is_rating

    stim = events.loc[is_stim].copy()
    if "condition" in stim:
        conds = (
            stim["condition"]
            .astype(str)
            .str.lower()
            .loc[lambda x: x.isin(["passive", "up", "down"])]
            .unique()
            .tolist()
        )
    else:
        tt = stim.get("trial_type", pd.Series("", index=stim.index)).astype(str).str.lower()
        labels = pd.Series("", index=stim.index)
        labels.loc[tt.str.contains("passive")] = "passive"
        labels.loc[tt.str.contains("up")] = "up"
        labels.loc[tt.str.contains("down")] = "down"
        conds = labels.loc[labels.isin(["passive", "up", "down"])].unique().tolist()

    if len(conds) != 1:
        raise RuntimeError(f"Could not infer one condition from {events_path}: {conds}")
    return conds[0], int(is_stim.sum()), int(is_rating.sum())


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    inv_path = args.inventory.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    inv = keep_inventory(pd.read_csv(inv_path, sep="\t"))
    problems: list[str] = []
    rows: list[dict[str, Any]] = []

    if len(inv) != 288:
        problems.append(f"Expected 288 QC-retained runs; found {len(inv)}")
    if inv["subject"].nunique() != 33:
        problems.append(f"Expected 33 subjects; found {inv['subject'].nunique()}")
    if inv.duplicated(["subject", "run"]).any():
        problems.append("Duplicate subject-run rows detected")

    for _, row in inv.iterrows():
        record: dict[str, Any] = {
            "subject": row["subject"],
            "run": int(row["run"]),
        }
        try:
            bold = resolve_path(row["bold_unsmoothed"], root)
            events = resolve_path(row["events_long"], root)
            confounds = resolve_path(row["confounds_glm"], root)
            record.update(
                bold_path=str(bold),
                events_path=str(events),
                confounds_path=str(confounds),
                bold_exists=bold.exists(),
                events_exists=events.exists(),
                confounds_exists=confounds.exists(),
            )
            if not bold.exists():
                problems.append(f"Missing BOLD: {bold}")
            if not events.exists():
                problems.append(f"Missing events: {events}")
            if not confounds.exists():
                problems.append(f"Missing confounds: {confounds}")
            if events.exists():
                condition, n_stim, n_rating = infer_condition(events)
                record.update(condition=condition, n_stim_events=n_stim, n_rating_events=n_rating)
                expected = 11 if condition == "passive" else 10
                if n_stim != expected or n_rating != expected:
                    problems.append(
                        f"Unexpected event count for {row['subject']} run-{int(row['run']):02d}: "
                        f"condition={condition}, stim={n_stim}, rating={n_rating}, expected={expected}"
                    )
        except Exception as exc:  # noqa: BLE001
            record["error"] = repr(exc)
            problems.append(f"{row['subject']} run-{int(row['run']):02d}: {exc}")
        rows.append(record)

    run_table = pd.DataFrame(rows)
    run_table.to_csv(outdir / "all_passive_dcm_preflight_runs.tsv", sep="\t", index=False)

    condition_counts = (
        run_table.get("condition", pd.Series(dtype=str)).value_counts().to_dict()
    )
    passive_per_subject = (
        run_table.loc[run_table.get("condition", pd.Series(dtype=str)).eq("passive")]
        .groupby("subject")
        .size()
        .value_counts()
        .sort_index()
        .to_dict()
    )

    expected_counts = {"passive": 222, "up": 33, "down": 33}
    if condition_counts != expected_counts:
        problems.append(
            f"Condition counts mismatch: found {condition_counts}; expected {expected_counts}"
        )

    subject_condition = (
        run_table.groupby(["subject", "condition"]).size().unstack(fill_value=0)
        if "condition" in run_table
        else pd.DataFrame()
    )
    for subject, values in subject_condition.iterrows():
        if int(values.get("up", 0)) != 1 or int(values.get("down", 0)) != 1:
            problems.append(f"{subject} does not have exactly one up and one down run")
        n_passive = int(values.get("passive", 0))
        if not 4 <= n_passive <= 7:
            problems.append(f"{subject} has {n_passive} passive runs; expected 4-7")

    summary = {
        "inventory": str(inv_path),
        "dataset_root": str(root),
        "n_qc_retained_runs": len(inv),
        "n_subjects": int(inv["subject"].nunique()),
        "condition_counts": condition_counts,
        "passive_runs_per_subject_distribution": {
            str(k): int(v) for k, v in passive_per_subject.items()
        },
        "all_bold_exist": bool(run_table.get("bold_exists", pd.Series([False])).all()),
        "all_events_exist": bool(run_table.get("events_exists", pd.Series([False])).all()),
        "all_confounds_exist": bool(run_table.get("confounds_exists", pd.Series([False])).all()),
        "problems": problems,
        "passed": not problems,
    }
    (outdir / "all_passive_dcm_preflight_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if problems:
        raise SystemExit("\nSessionwise DCM input preflight failed.")
    print("\nSessionwise DCM input preflight passed.")


if __name__ == "__main__":
    main()
