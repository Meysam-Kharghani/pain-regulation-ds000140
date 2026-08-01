#!/usr/bin/env python3
"""
Validate computational pain-regulation success definitions.

This script checks that up/down regulation success was computed in the intended,
non-circular direction relative to the passive-only expected-pain model:
    up success   = observed rating - expected passive pain
    down success = expected passive pain - observed rating

It does not refit the model; it audits the trialwise predictions produced by
validate_regulation_success.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trialwise", required=True, help="trialwise_model_predictions.tsv")
    p.add_argument("--outdir", required=True, help="Output directory for QC files")
    p.add_argument("--tol", type=float, default=1e-6)
    return p.parse_args()


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 3:
        return np.nan
    return float(np.corrcoef(x.iloc[:, 0], x.iloc[:, 1])[0, 1])


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.trialwise, sep="\t")
    required = [
        "subject", "condition", "is_up", "is_down", "is_reg",
        "rating_z_subject", "expected_pain_z_passive_model", "reg_success_z_model",
        "rating_raw", "expected_pain_raw_passive_model", "reg_success_raw_model",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    reg = df[df["is_reg"].astype(int).eq(1)].copy()
    reg = reg.replace([np.inf, -np.inf], np.nan)

    # Intended formula checks.
    reg["expected_formula_z"] = np.nan
    up = reg["is_up"].astype(int).eq(1)
    down = reg["is_down"].astype(int).eq(1)
    reg.loc[up, "expected_formula_z"] = (
        reg.loc[up, "rating_z_subject"] - reg.loc[up, "expected_pain_z_passive_model"]
    )
    reg.loc[down, "expected_formula_z"] = (
        reg.loc[down, "expected_pain_z_passive_model"] - reg.loc[down, "rating_z_subject"]
    )
    reg["formula_error_z"] = reg["reg_success_z_model"] - reg["expected_formula_z"]

    reg["expected_formula_raw"] = np.nan
    reg.loc[up, "expected_formula_raw"] = (
        reg.loc[up, "rating_raw"] - reg.loc[up, "expected_pain_raw_passive_model"]
    )
    reg.loc[down, "expected_formula_raw"] = (
        reg.loc[down, "expected_pain_raw_passive_model"] - reg.loc[down, "rating_raw"]
    )
    reg["formula_error_raw"] = reg["reg_success_raw_model"] - reg["expected_formula_raw"]

    rows: List[Dict[str, object]] = []
    for label, subset in [("all_reg", reg), ("up", reg[up]), ("down", reg[down])]:
        rows.append({
            "subset": label,
            "n_trials": int(len(subset)),
            "n_subjects": int(subset["subject"].nunique()),
            "max_abs_formula_error_z": float(np.nanmax(np.abs(subset["formula_error_z"]))) if len(subset) else np.nan,
            "mean_abs_formula_error_z": float(np.nanmean(np.abs(subset["formula_error_z"]))) if len(subset) else np.nan,
            "max_abs_formula_error_raw": float(np.nanmax(np.abs(subset["formula_error_raw"]))) if len(subset) else np.nan,
            "corr_success_with_observed_rating_z": safe_corr(subset["reg_success_z_model"], subset["rating_z_subject"]),
            "corr_success_with_expected_pain_z": safe_corr(subset["reg_success_z_model"], subset["expected_pain_z_passive_model"]),
            "mean_success_z": float(np.nanmean(subset["reg_success_z_model"])),
            "sd_success_z": float(np.nanstd(subset["reg_success_z_model"], ddof=1)),
        })

    qc = pd.DataFrame(rows)
    qc["formula_pass_z"] = qc["max_abs_formula_error_z"].le(args.tol)
    qc.to_csv(outdir / "tables" / "success_definition_qc.tsv", sep="\t", index=False)

    cols = [
        "subject", "condition", "is_up", "is_down", "rating_z_subject",
        "expected_pain_z_passive_model", "reg_success_z_model", "expected_formula_z",
        "formula_error_z", "rating_raw", "expected_pain_raw_passive_model",
        "reg_success_raw_model", "expected_formula_raw", "formula_error_raw",
    ]
    reg[cols].to_csv(outdir / "tables" / "trialwise_success_formula_audit.tsv", sep="\t", index=False)

    manifest = {
        "trialwise": str(Path(args.trialwise).resolve()),
        "n_trials_total": int(len(df)),
        "n_reg_trials": int(len(reg)),
        "n_subjects": int(df["subject"].nunique()),
        "definition": {
            "up_success": "observed rating minus passive-expected pain",
            "down_success": "passive-expected pain minus observed rating",
            "positive_success": "rating moved in the instructed direction relative to passive expected pain",
        },
        "all_formula_checks_passed": bool(qc["formula_pass_z"].all()),
    }
    with open(outdir / "manifest_success_definition_qc.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Output file:", outdir / "tables" / "success_definition_qc.tsv")
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
