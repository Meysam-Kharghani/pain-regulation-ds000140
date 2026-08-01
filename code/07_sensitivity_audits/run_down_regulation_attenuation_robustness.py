#!/usr/bin/env python3
"""
Robustness audit for down-regulation attenuation of sensory ROI and pain-signature expression.

Purpose
-------
This analysis summarizes down-regulation attenuation across the main and
motion-sensitivity analysis sets. It evaluates two result classes:

- down-regulation versus passive stimulation in sensory ROI expression;
- down-regulation versus passive stimulation in pain-signature expression.

This script searches for ROI/signature statistics and subject-level extraction
tables, summarizes robustness across analysis sets, and produces:
- down_minus_passive ROI/signature result tables;
- sign-stability summaries across main and sensitivity sets;
- a compact summary table;
- optional paired subject-level sensitivity if extraction tables are available.

Example
-------
PROJECT=${PROJECT_ROOT}

python run_down_regulation_attenuation_robustness.py \
  --project "$PROJECT" \
  --outdir "$PROJECT/derivatives/sensitivity_audits/down_regulation_attenuation_robustness"

If automatic discovery fails, pass explicit stats tables:
python run_down_regulation_attenuation_robustness.py \
  --roi-stats path/to/roi_stats.tsv \
  --signature-stats path/to/signature_stats.tsv \
  --outdir out
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


EFFECT_TERMS = ("down_minus_passive", "down-regulation versus passive", "down_vs_passive",
                "downminuspassive", "down_passive")
SENSORY_TERMS = ("s1", "s2", "operculum", "posterior_insula", "whole_insula", "insula",
                 "thalamus", "dacc", "mcc", "sensory")
SIGNATURE_TERMS = ("nps", "geuter", "siips", "pain", "cpm", "cpdm", "mpa")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def read_table(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() in [".tsv", ".tab"]:
            return pd.read_csv(path, sep="\t")
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path)
    except Exception:
        return None
    return None


def iter_candidate_files(project: Optional[Path]) -> Iterable[Path]:
    if project is None:
        return []
    roots = [project / "derivatives" / "sensitivity_audits", project / "derivatives", project]
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.tsv", "*.csv", "*.xlsx"):
            for p in root.rglob(ext):
                ps = str(p)
                if ps in seen:
                    continue
                seen.add(ps)
                name = norm(ps)
                if any(t in name for t in ("roi", "signature", "nps", "fixed_effect", "stats", "extract")):
                    yield p


def guess_kind(path: Path, df: pd.DataFrame) -> str:
    p = norm(str(path))
    cols = " ".join(norm(c) for c in df.columns)
    if "signature" in p or "signature" in cols or "nps" in cols:
        return "signature"
    if "roi" in p or "roi" in cols:
        return "roi"
    return "unknown"


def has_stat_columns(df: pd.DataFrame) -> bool:
    cols = [norm(c) for c in df.columns]
    return any(c in cols or "mean" == c for c in cols) and any("p" == c or c.endswith("_p") or "p_value" in c for c in cols)


def find_col(df: pd.DataFrame, options: Sequence[str], contains: bool = True) -> Optional[str]:
    for opt in options:
        for c in df.columns:
            cn = norm(c)
            if (contains and opt in cn) or ((not contains) and opt == cn):
                return c
    return None


def standardize_stats(df: pd.DataFrame, kind: str, source_path: str) -> pd.DataFrame:
    out = df.copy()
    # Common column guesses
    effect_col = find_col(out, ["effect", "contrast", "term"])
    set_col = find_col(out, ["analysis_set", "set", "subset"])
    family_col = find_col(out, ["family", "roi_family", "signature_family"])
    roi_col = find_col(out, ["roi", "mask", "region", "signature", "pattern"])
    sig_col = find_col(out, ["signature", "pattern"])
    name_col = sig_col if kind == "signature" and sig_col is not None else roi_col

    mean_col = find_col(out, ["mean"], contains=False) or find_col(out, ["mean"])
    t_col = find_col(out, ["t", "statistic", "stat"], contains=False) or find_col(out, ["t_"])
    p_col = find_col(out, ["p", "p_value", "pval"], contains=False) or find_col(out, ["p_value", "pval"])
    q_col = find_col(out, ["q", "q_family", "q_fdr", "fdr"], contains=False) or find_col(out, ["q_family", "q_fdr", "fdr"])

    rows = pd.DataFrame({
        "kind": kind,
        "source_path": source_path,
        "analysis_set": out[set_col] if set_col else "unknown",
        "effect": out[effect_col] if effect_col else "unknown",
        "family": out[family_col] if family_col else "unknown",
        "name": out[name_col] if name_col else out.index.astype(str),
        "mean": pd.to_numeric(out[mean_col], errors="coerce") if mean_col else np.nan,
        "stat": pd.to_numeric(out[t_col], errors="coerce") if t_col else np.nan,
        "p": pd.to_numeric(out[p_col], errors="coerce") if p_col else np.nan,
        "q": pd.to_numeric(out[q_col], errors="coerce") if q_col else np.nan,
    })
    return rows


def is_down_minus_passive(effect) -> bool:
    e = norm(effect)
    return ("down" in e and "passive" in e and ("minus" in e or "vs" in e or "reg" in e)) or any(t in e for t in EFFECT_TERMS)


def is_sensory_name(name, family="") -> bool:
    n = norm(name) + "_" + norm(family)
    return any(t in n for t in SENSORY_TERMS)


def is_pain_signature(name) -> bool:
    n = norm(name)
    return any(t in n for t in SIGNATURE_TERMS)


def bh_q(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    pv = p[mask]
    if len(pv) == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    q[mask] = out
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=None)
    ap.add_argument("--roi-stats", type=Path, nargs="*", default=[])
    ap.add_argument("--signature-stats", type=Path, nargs="*", default=[])
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    files = []
    explicit = [(p, "roi") for p in args.roi_stats] + [(p, "signature") for p in args.signature_stats]
    for p, kind in explicit:
        if p.exists():
            files.append((p, kind))
    if args.project is not None:
        for p in iter_candidate_files(args.project):
            df = read_table(p)
            if df is None or df.empty:
                continue
            kind = guess_kind(p, df)
            if kind != "unknown":
                files.append((p, kind))

    # Deduplicate
    seen = set()
    dedup = []
    for p, kind in files:
        key = (str(p), kind)
        if key not in seen:
            seen.add(key)
            dedup.append((p, kind))
    files = dedup

    candidate_rows = []
    standardized = []
    for p, forced_kind in files:
        df = read_table(p)
        if df is None or df.empty:
            continue
        kind = forced_kind if forced_kind != "unknown" else guess_kind(p, df)
        candidate_rows.append({
            "path": str(p),
            "kind": kind,
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "columns": ";".join(map(str, df.columns[:50])),
        })
        if has_stat_columns(df):
            standardized.append(standardize_stats(df, kind, str(p)))

    cand_df = pd.DataFrame(candidate_rows)
    cand_df.to_csv(args.outdir / "candidate_roi_signature_tables.tsv", sep="\t", index=False)

    if not standardized:
        (args.outdir / "README_no_stats_found.md").write_text(
            "No ROI/signature statistics tables were automatically standardized. "
            "Open candidate_roi_signature_tables.tsv, identify the ROI/signature stats tables, "
            "and rerun with --roi-stats and --signature-stats.\n",
            encoding="utf-8")
        print(f"[WARN] No stats standardized. See {args.outdir}")
        return

    all_stats = pd.concat(standardized, ignore_index=True)
    all_stats["is_down_minus_passive"] = all_stats["effect"].map(is_down_minus_passive)
    all_stats["is_sensory_roi"] = [is_sensory_name(n, f) for n, f in zip(all_stats["name"], all_stats["family"])]
    all_stats["is_pain_signature"] = all_stats["name"].map(is_pain_signature)
    all_stats["negative"] = all_stats["mean"] < 0
    # Compute Benjamini-Hochberg adjusted values within kind, source, effect, family, and analysis set when adjusted values are unavailable.
    if all_stats["q"].isna().all():
        all_stats["q"] = all_stats.groupby(["kind", "source_path", "analysis_set", "effect", "family"])["p"].transform(bh_q)

    all_stats.to_csv(args.outdir / "all_standardized_roi_signature_stats.tsv", sep="\t", index=False)

    target = all_stats[
        all_stats["is_down_minus_passive"] &
        (
            ((all_stats["kind"] == "roi") & all_stats["is_sensory_roi"]) |
            ((all_stats["kind"] == "signature") & all_stats["is_pain_signature"])
        )
    ].copy()

    target = target.sort_values(["kind", "analysis_set", "q", "p", "name"], na_position="last")
    target.to_csv(args.outdir / "down_minus_passive_target_results.tsv", sep="\t", index=False)

    # Robustness summaries.
    sig = target[np.isfinite(target["q"]) & (target["q"] < 0.05)].copy()
    sig_neg = sig[sig["negative"]].copy()

    summary = []
    for (kind, name), g in target.groupby(["kind", "name"], dropna=False):
        sets = sorted(set(map(str, g["analysis_set"])))
        n_sets = len(sets)
        n_neg = int((g["mean"] < 0).sum())
        n_q05 = int(((g["q"] < 0.05) & np.isfinite(g["q"])).sum())
        n_q05_neg = int(((g["q"] < 0.05) & np.isfinite(g["q"]) & (g["mean"] < 0)).sum())
        best = g.sort_values(["q", "p"], na_position="last").iloc[0]
        summary.append({
            "kind": kind,
            "name": name,
            "n_analysis_sets_seen": n_sets,
            "analysis_sets": ";".join(sets),
            "negative_in_n_sets": n_neg,
            "q_lt_0_05_in_n_sets": n_q05,
            "q_lt_0_05_and_negative_in_n_sets": n_q05_neg,
            "best_analysis_set": best["analysis_set"],
            "best_mean": best["mean"],
            "best_stat": best["stat"],
            "best_p": best["p"],
            "best_q": best["q"],
            "robust_negative_all_sets": n_sets > 0 and n_neg == n_sets,
        })
    rob = pd.DataFrame(summary).sort_values(["kind", "q_lt_0_05_and_negative_in_n_sets", "negative_in_n_sets", "best_q"], ascending=[True, False, False, True])
    rob.to_csv(args.outdir / "down_minus_passive_sign_stability_summary.tsv", sep="\t", index=False)

    # Compact main-analysis summary: main set first if available; otherwise best set.
    main_like = target[target["analysis_set"].astype(str).str.lower().isin(["main", "main_analysis", "main_set"])]
    compact_source = main_like if len(main_like) else target
    compact = compact_source[
        (compact_source["negative"]) & (compact_source["q"] < 0.05)
    ].copy()
    compact = compact.sort_values(["kind", "q", "p", "name"])
    compact[["kind", "analysis_set", "name", "family", "mean", "stat", "p", "q", "source_path"]].to_csv(
        args.outdir / "compact_down_attenuation_table.tsv", sep="\t", index=False
    )

    # README interpretation
    lines = []
    lines.append("# Down-regulation attenuation robustness\n\n")
    lines.append("Primary files:\n")
    lines.append("- `down_minus_passive_target_results.tsv`: all down-minus-passive sensory ROI and pain-signature rows found.\n")
    lines.append("- `down_minus_passive_sign_stability_summary.tsv`: sign and q<0.05 stability across analysis sets.\n")
    lines.append("- `compact_down_attenuation_table.tsv`: compact negative q<0.05 rows, preferring the main analysis set.\n\n")
    lines.append("Interpretation guidance:\n")
    lines.append("- Evidence for down-regulation attenuation is strongest when the same sensory ROIs or signatures are negative in the main set and remain negative across sensitivity sets.\n")
    lines.append("- Effects surviving correction only in the main set, with stable signs in sensitivity sets, support a qualified interpretation based on corrected main analyses and sign-consistent sensitivity results.\n")
    lines.append("- NPS results derived from smoothed GLM maps are interpreted at the signature-expression level rather than as evidence for a spatially fine-grained neural mechanism.\n\n")
    lines.append(f"Found {len(target)} target rows and {len(compact)} compact significant negative rows.\n")
    (args.outdir / "README.md").write_text("".join(lines), encoding="utf-8")

    print(f"[OK] Outputs written to {args.outdir}")
    print(f"[OK] Target rows: {len(target)} | compact significant negative rows: {len(compact)}")


if __name__ == "__main__":
    main()
