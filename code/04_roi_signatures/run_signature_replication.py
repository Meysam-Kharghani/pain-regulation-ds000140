#!/usr/bin/env python3
"""
run_signature_replication.py

Purpose
-------
Run a replication-style NPS analysis for comparison with the original ds000140 report.

This script uses the already extracted NPS expression values from:
  roi_signature_analysis_with_nps/tables/signature_values.tsv

It does not re-extract signatures from images. It tests whether the NPS shows:
  1) strong stimulus/temperature sensitivity,
  2) no canonical up-vs-down self-regulation effect,
  3) a possible down-vs-passive reduction, which is the new/divergent finding.

Important interpretation
------------------------
The original paper used single-trial NPS dot-products in multilevel models.
This script uses subject-level first-level GLM effects/contrasts. Therefore,
it is a "replication-style" analysis, not an exact reproduction of their
trial-level model.

Inputs
------
--signature-values:
  .../roi_signature_analysis_with_nps/tables/signature_values.tsv

Outputs
-------
outdir/
  manifest.json
  nps_original_style_replication_report.md
  tables/
    nps_direct_effect_tests.tsv
    nps_condition_paired_tests.tsv
    nps_temperature_slope_tests.tsv
    nps_original_style_summary.tsv
    nps_subject_condition_values.tsv
  figures/
    fig_nps_condition_means_<measure>_<set>.png
    fig_nps_direct_contrasts_<measure>_<set>.png
    fig_nps_temperature_slopes_<measure>_<set>.png

Example
-------
python run_signature_replication.py \
  --signature-values <PROJECT_ROOT>/derivatives/roi_signature_analysis_with_nps/tables/signature_values.tsv \
  --outdir <PROJECT_ROOT>/derivatives/nps_original_style_replication_analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_seed(*parts: object) -> int:
    """Return a process-independent 32-bit seed from semantic labels."""
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def save_tsv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=False)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_num(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def cohen_dz(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 1e-12 else np.nan


def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 5000, seed: int = 1) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def sign_flip_p(x: np.ndarray, n_perm: int = 20000, seed: int = 1) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    obs = abs(x.mean())
    rng = np.random.default_rng(seed)
    sims = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=len(x), replace=True)
        sims[i] = abs((x * signs).mean())
    return float((np.sum(sims >= obs) + 1) / (n_perm + 1))


def one_sample_test(values: Sequence[float], label: str, seed: int = 1) -> Dict[str, object]:
    x = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    out = {"test": label, "n": int(len(x))}
    if len(x) < 2:
        out.update({
            "mean": np.nan, "sd": np.nan, "se": np.nan,
            "t": np.nan, "p_t": np.nan, "p_signflip": np.nan,
            "dz": np.nan, "ci_lo_boot": np.nan, "ci_hi_boot": np.nan,
            "wilcoxon_stat": np.nan, "wilcoxon_p": np.nan,
        })
        return out

    tt = stats.ttest_1samp(x, 0.0)
    lo, hi = bootstrap_ci_mean(x, seed=seed)
    out.update({
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "se": float(x.std(ddof=1) / np.sqrt(len(x))),
        "t": float(tt.statistic),
        "p_t": float(tt.pvalue),
        "p_signflip": sign_flip_p(x, seed=seed),
        "dz": cohen_dz(x),
        "ci_lo_boot": lo,
        "ci_hi_boot": hi,
    })
    try:
        w = stats.wilcoxon(x)
        out["wilcoxon_stat"] = float(w.statistic)
        out["wilcoxon_p"] = float(w.pvalue)
    except Exception:
        out["wilcoxon_stat"] = np.nan
        out["wilcoxon_p"] = np.nan
    return out


def fdr_by_group(df: pd.DataFrame, group_cols: List[str], p_col: str, q_col: str) -> pd.DataFrame:
    out = df.copy()
    out[q_col] = np.nan
    if out.empty or p_col not in out.columns:
        return out

    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        p = pd.to_numeric(out.loc[idx, p_col], errors="coerce")
        mask = p.notna() & np.isfinite(p)
        if mask.sum() > 0:
            q = multipletests(p[mask].to_numpy(), method="fdr_bh")[1]
            out.loc[p[mask].index, q_col] = q
    return out


def fmt(x, digits=4) -> str:
    try:
        if pd.isna(x):
            return "NA"
        if abs(float(x)) < 1e-4 and float(x) != 0:
            return f"{float(x):.2e}"
        return f"{float(x):.{digits}g}"
    except Exception:
        return str(x)


# =============================================================================
# Core mappings
# =============================================================================

COND_EFFECTS = {
    "passive": "stim_passive_FE",
    "up": "stim_up_FE",
    "down": "stim_down_FE",
}

TEMP_EFFECTS = {
    "passive": "stim_passive_temp_FE",
    "up": "stim_up_temp_FE",
    "down": "stim_down_temp_FE",
}

DIRECT_CONTRASTS = [
    "up_minus_down_FE",
    "up_minus_passive_FE",
    "down_minus_passive_FE",
    "regulation_mean_minus_passive_FE",
]

CANONICAL_EFFECTS = list(COND_EFFECTS.values()) + list(TEMP_EFFECTS.values()) + DIRECT_CONTRASTS


# =============================================================================
# Loading
# =============================================================================

def load_signature_values(path: Path, signature: str, sets: List[str] | None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = ["set_name", "subject", "effect", "signature", "status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"signature_values file is missing required columns: {missing}")

    df = df.copy()
    df["signature"] = df["signature"].astype(str)
    df = df[(df["signature"] == signature) & (df["status"] == "OK")].copy()

    if sets is not None and len(sets) > 0:
        df = df[df["set_name"].isin(sets)].copy()

    for m in ["signature_weighted_mean", "signature_dot", "signature_cosine"]:
        if m in df.columns:
            df[m] = safe_num(df[m])

    return df.reset_index(drop=True)


# =============================================================================
# Analyses
# =============================================================================

def direct_effect_tests(df: pd.DataFrame, measures: List[str]) -> pd.DataFrame:
    rows = []
    for (set_name, effect), d in df.groupby(["set_name", "effect"]):
        for measure in measures:
            if measure not in d.columns:
                continue
            row = one_sample_test(d[measure], effect, seed=stable_seed(set_name, effect, measure))
            row.update({
                "analysis": "direct_effect_one_sample",
                "set_name": set_name,
                "effect": effect,
                "measure": measure,
                "signature": d["signature"].iloc[0],
            })
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = fdr_by_group(out, ["set_name", "measure"], "p_t", "q_fdr_within_set_measure")
        cols = ["analysis", "set_name", "signature", "measure", "effect", "test", "n", "mean", "sd", "se", "t", "p_t",
                "q_fdr_within_set_measure", "p_signflip", "wilcoxon_p", "dz", "ci_lo_boot", "ci_hi_boot"]
        out = out[[c for c in cols if c in out.columns]]
    return out


def build_condition_wide(df: pd.DataFrame, measure: str) -> pd.DataFrame:
    d = df[df["effect"].isin(CANONICAL_EFFECTS)].copy()
    d = d[["set_name", "subject", "effect", measure]].dropna()
    wide = d.pivot_table(index=["set_name", "subject"], columns="effect", values=measure, aggfunc="mean").reset_index()
    wide.columns.name = None

    # Rename condition amplitudes
    for cond, eff in COND_EFFECTS.items():
        if eff in wide.columns:
            wide[f"cond_{cond}"] = wide[eff]
    for cond, eff in TEMP_EFFECTS.items():
        if eff in wide.columns:
            wide[f"temp_{cond}"] = wide[eff]

    # Derived paired contrasts from condition amplitudes
    if {"cond_up", "cond_down"}.issubset(wide.columns):
        wide["paired_up_minus_down"] = wide["cond_up"] - wide["cond_down"]
    if {"cond_up", "cond_passive"}.issubset(wide.columns):
        wide["paired_up_minus_passive"] = wide["cond_up"] - wide["cond_passive"]
    if {"cond_down", "cond_passive"}.issubset(wide.columns):
        wide["paired_down_minus_passive"] = wide["cond_down"] - wide["cond_passive"]
    if {"cond_up", "cond_down", "cond_passive"}.issubset(wide.columns):
        wide["paired_regulation_mean_minus_passive"] = 0.5 * (wide["cond_up"] + wide["cond_down"]) - wide["cond_passive"]

    # Derived paired contrasts from temperature slopes
    if {"temp_up", "temp_down"}.issubset(wide.columns):
        wide["temp_up_minus_down"] = wide["temp_up"] - wide["temp_down"]
    if {"temp_up", "temp_passive"}.issubset(wide.columns):
        wide["temp_up_minus_passive"] = wide["temp_up"] - wide["temp_passive"]
    if {"temp_down", "temp_passive"}.issubset(wide.columns):
        wide["temp_down_minus_passive"] = wide["temp_down"] - wide["temp_passive"]
    if {"temp_up", "temp_down", "temp_passive"}.issubset(wide.columns):
        wide["temp_regulation_mean_minus_passive"] = 0.5 * (wide["temp_up"] + wide["temp_down"]) - wide["temp_passive"]

    wide["measure"] = measure
    return wide


def condition_paired_tests(wide_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    paired_cols = [
        "paired_up_minus_down",
        "paired_up_minus_passive",
        "paired_down_minus_passive",
        "paired_regulation_mean_minus_passive",
    ]
    cond_cols = ["cond_passive", "cond_up", "cond_down"]

    for (set_name, measure), d in wide_all.groupby(["set_name", "measure"]):
        for col in cond_cols + paired_cols:
            if col not in d.columns:
                continue
            row = one_sample_test(d[col], col, seed=stable_seed(set_name, measure, col))
            row.update({
                "analysis": "condition_level_or_paired",
                "set_name": set_name,
                "measure": measure,
                "quantity": col,
            })
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = fdr_by_group(out, ["set_name", "measure"], "p_t", "q_fdr_within_set_measure")
    return out


def temperature_slope_tests(wide_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    temp_cols = [
        "temp_passive",
        "temp_up",
        "temp_down",
        "temp_up_minus_down",
        "temp_up_minus_passive",
        "temp_down_minus_passive",
        "temp_regulation_mean_minus_passive",
    ]

    for (set_name, measure), d in wide_all.groupby(["set_name", "measure"]):
        for col in temp_cols:
            if col not in d.columns:
                continue
            row = one_sample_test(d[col], col, seed=stable_seed(set_name, measure, col))
            row.update({
                "analysis": "temperature_slope_or_slope_difference",
                "set_name": set_name,
                "measure": measure,
                "quantity": col,
            })
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = fdr_by_group(out, ["set_name", "measure"], "p_t", "q_fdr_within_set_measure")
    return out


def original_style_summary(direct: pd.DataFrame, condition: pd.DataFrame, temp: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (set_name, measure), d0 in direct.groupby(["set_name", "measure"]):
        def get_effect(effect):
            d = d0[d0["effect"] == effect]
            return None if d.empty else d.iloc[0].to_dict()

        def get_quantity(df, quantity):
            d = df[(df["set_name"] == set_name) & (df["measure"] == measure) & (df["quantity"] == quantity)]
            return None if d.empty else d.iloc[0].to_dict()

        passive_temp = get_effect("stim_passive_temp_FE")
        up_down = get_effect("up_minus_down_FE")
        up_passive = get_effect("up_minus_passive_FE")
        down_passive = get_effect("down_minus_passive_FE")
        regulation_mean_passive = get_effect("regulation_mean_minus_passive_FE")

        # Direct contrast tests are preferred because they come from the GLM contrast maps.
        temp_p = passive_temp.get("p_t", np.nan) if passive_temp else np.nan
        temp_mean = passive_temp.get("mean", np.nan) if passive_temp else np.nan
        ud_p = up_down.get("p_t", np.nan) if up_down else np.nan
        ud_mean = up_down.get("mean", np.nan) if up_down else np.nan
        dp_p = down_passive.get("p_t", np.nan) if down_passive else np.nan
        dp_mean = down_passive.get("mean", np.nan) if down_passive else np.nan
        up_p = up_passive.get("p_t", np.nan) if up_passive else np.nan
        rp_p = regulation_mean_passive.get("p_t", np.nan) if regulation_mean_passive else np.nan

        original_like = bool((pd.notna(temp_p) and temp_p < 0.05 and temp_mean > 0) and
                             (pd.notna(ud_p) and ud_p >= 0.05))
        down_specific_divergence = bool(pd.notna(dp_p) and dp_p < 0.05 and pd.notna(dp_mean) and dp_mean < 0)
        broad_self_reg_nps = bool(
            (pd.notna(ud_p) and ud_p < 0.05) or
            (pd.notna(up_p) and up_p < 0.05) or
            (pd.notna(rp_p) and rp_p < 0.05)
        )

        rows.append({
            "set_name": set_name,
            "measure": measure,
            "signature": "NPS",
            "passive_temperature_mean": temp_mean,
            "passive_temperature_p": temp_p,
            "up_minus_down_mean": ud_mean,
            "up_minus_down_p": ud_p,
            "up_minus_passive_mean": up_passive.get("mean", np.nan) if up_passive else np.nan,
            "up_minus_passive_p": up_p,
            "down_minus_passive_mean": dp_mean,
            "down_minus_passive_p": dp_p,
            "regulation_mean_minus_passive_mean": regulation_mean_passive.get("mean", np.nan) if regulation_mean_passive else np.nan,
            "regulation_mean_minus_passive_p": rp_p,
            "original_like_pattern": original_like,
            "down_specific_divergence": down_specific_divergence,
            "broad_self_regulation_effect_on_NPS": broad_self_reg_nps,
            "interpretation": (
                "Original-like on canonical up-vs-down test; also shows down-specific NPS reduction"
                if original_like and down_specific_divergence else
                "Original-like on canonical up-vs-down test"
                if original_like else
                "Canonical up-vs-down test differs from original-style null"
                if pd.notna(ud_p) and ud_p < 0.05 else
                "Inconclusive"
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Figures
# =============================================================================

def mean_ci(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return np.nan, np.nan
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def plot_condition_means(wide: pd.DataFrame, measure: str, set_name: str, outpath: Path):
    d = wide[(wide["measure"] == measure) & (wide["set_name"] == set_name)].copy()
    cols = ["cond_passive", "cond_up", "cond_down"]
    labels = ["passive", "up", "down"]
    vals, errs = [], []
    for c in cols:
        m, e = mean_ci(d, c)
        vals.append(m)
        errs.append(e)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(np.arange(len(vals)), vals, yerr=errs, fmt="o", capsize=3)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(measure)
    ax.set_title(f"NPS condition amplitudes ({set_name})")
    fig.tight_layout()
    ensure_dir(outpath.parent)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_direct_contrasts(direct: pd.DataFrame, measure: str, set_name: str, outpath: Path):
    d = direct[(direct["measure"] == measure) & (direct["set_name"] == set_name) &
               (direct["effect"].isin(DIRECT_CONTRASTS))].copy()
    if d.empty:
        return
    d = d.set_index("effect").reindex(DIRECT_CONTRASTS).reset_index()
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(x, d["mean"], yerr=1.96 * d["se"], fmt="o", capsize=3)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(d["effect"], rotation=25, ha="right")
    ax.set_ylabel(measure)
    ax.set_title(f"NPS direct GLM contrasts ({set_name})")
    fig.tight_layout()
    ensure_dir(outpath.parent)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_temperature_slopes(wide: pd.DataFrame, measure: str, set_name: str, outpath: Path):
    d = wide[(wide["measure"] == measure) & (wide["set_name"] == set_name)].copy()
    cols = ["temp_passive", "temp_up", "temp_down"]
    labels = ["passive temp", "up temp", "down temp"]
    vals, errs = [], []
    for c in cols:
        m, e = mean_ci(d, c)
        vals.append(m)
        errs.append(e)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(np.arange(len(vals)), vals, yerr=errs, fmt="o", capsize=3)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(measure)
    ax.set_title(f"NPS temperature slopes ({set_name})")
    fig.tight_layout()
    ensure_dir(outpath.parent)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# =============================================================================
# Report
# =============================================================================

def write_report(outdir: Path, manifest: Dict, summary: pd.DataFrame, direct: pd.DataFrame,
                 condition: pd.DataFrame, temp: pd.DataFrame, primary_set: str, primary_measure: str):
    lines = []
    lines.append("# NPS original-style replication report\n\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis tests whether the updated NPS results reproduce the original paper's core pattern: "
        "strong NPS sensitivity to stimulus/temperature, but no canonical self-regulation effect in the "
        "up-versus-down contrast. It also explicitly tests the new down-minus-passive finding.\n\n"
    )
    lines.append("## Inputs\n")
    lines.append(f"- Signature values: `{manifest['signature_values']}`\n")
    lines.append(f"- Signature: `{manifest['signature']}`\n")
    lines.append(f"- Measures: `{', '.join(manifest['measures'])}`\n")
    lines.append(f"- Sets: `{', '.join(manifest['sets'])}`\n\n")
    lines.append("## Important limitation\n")
    lines.append(
        "The original study used single-trial NPS dot-products in multilevel models. "
        "This script uses subject-level first-level GLM effect/contrast maps, so it is a replication-style "
        "analysis rather than an exact reproduction of the original trial-level model.\n\n"
    )

    s = summary[(summary["set_name"] == primary_set) & (summary["measure"] == primary_measure)]
    if not s.empty:
        r = s.iloc[0]
        lines.append(f"## Primary summary: {primary_set}, {primary_measure}\n")
        lines.append(f"- Passive temperature NPS effect: mean={fmt(r['passive_temperature_mean'])}, p={fmt(r['passive_temperature_p'])}\n")
        lines.append(f"- Canonical up-minus-down NPS effect: mean={fmt(r['up_minus_down_mean'])}, p={fmt(r['up_minus_down_p'])}\n")
        lines.append(f"- Up-minus-passive NPS effect: mean={fmt(r['up_minus_passive_mean'])}, p={fmt(r['up_minus_passive_p'])}\n")
        lines.append(f"- Down-minus-passive NPS effect: mean={fmt(r['down_minus_passive_mean'])}, p={fmt(r['down_minus_passive_p'])}\n")
        lines.append(f"- Regmean-minus-passive NPS effect: mean={fmt(r['regulation_mean_minus_passive_mean'])}, p={fmt(r['regulation_mean_minus_passive_p'])}\n")
        lines.append(f"- Original-like canonical pattern: `{r['original_like_pattern']}`\n")
        lines.append(f"- Down-specific divergence: `{r['down_specific_divergence']}`\n")
        lines.append(f"- Interpretation: {r['interpretation']}\n\n")

    lines.append("## Interpretation\n")
    lines.append(
        "The subject-level contrast-map path is consistent with a null canonical up-minus-down NPS effect while "
        "showing a negative down-minus-passive estimate. The rating-adjusted FWL-LSS analysis reported under "
        "results/reporting does not reproduce the down-minus-passive effect. Together, these results indicate analysis-path "
        "dependence rather than a general NPS effect of regulation.\n\n"
    )

    lines.append("## Output tables\n")
    lines.append("- `tables/nps_direct_effect_tests.tsv`\n")
    lines.append("- `tables/nps_condition_paired_tests.tsv`\n")
    lines.append("- `tables/nps_temperature_slope_tests.tsv`\n")
    lines.append("- `tables/nps_original_style_summary.tsv`\n")
    lines.append("- `tables/nps_subject_condition_values.tsv`\n")

    (outdir / "nps_original_style_replication_report.md").write_text("".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signature-values", required=True, help="Path to signature_values.tsv from roi_signature_analysis_with_nps")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--signature", default="NPS")
    parser.add_argument("--sets", default="main,fd_bad_run_exclude,high_motion_subject_exclude",
                        help="Comma-separated set names, or ALL")
    parser.add_argument("--measures", default="signature_weighted_mean,signature_dot",
                        help="Comma-separated measures to test")
    parser.add_argument("--primary-set", default="main")
    parser.add_argument("--primary-measure", default="signature_weighted_mean")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sig_path = Path(args.signature_values).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    if not sig_path.exists():
        raise FileNotFoundError(f"signature-values not found: {sig_path}")

    if outdir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(outdir)
    ensure_dir(outdir)
    ensure_dir(outdir / "tables")
    ensure_dir(outdir / "figures")

    sets = None if args.sets.strip().upper() == "ALL" else [s.strip() for s in args.sets.split(",") if s.strip()]
    measures = [m.strip() for m in args.measures.split(",") if m.strip()]

    df = load_signature_values(sig_path, args.signature, sets)
    if df.empty:
        raise ValueError(f"No OK rows found for signature={args.signature} and sets={sets}")

    available_measures = [m for m in measures if m in df.columns]
    if not available_measures:
        raise ValueError(f"None of requested measures are available. Requested={measures}, columns={df.columns.tolist()}")

    direct = direct_effect_tests(df, available_measures)
    save_tsv(direct, outdir / "tables" / "nps_direct_effect_tests.tsv")

    wide_list = [build_condition_wide(df, m) for m in available_measures]
    wide = pd.concat(wide_list, ignore_index=True)
    save_tsv(wide, outdir / "tables" / "nps_subject_condition_values.tsv")

    condition = condition_paired_tests(wide)
    save_tsv(condition, outdir / "tables" / "nps_condition_paired_tests.tsv")

    temp = temperature_slope_tests(wide)
    save_tsv(temp, outdir / "tables" / "nps_temperature_slope_tests.tsv")

    summary = original_style_summary(direct, condition, temp)
    save_tsv(summary, outdir / "tables" / "nps_original_style_summary.tsv")

    # Figures for primary set/each measure
    for m in available_measures:
        for s in sorted(df["set_name"].unique()):
            plot_condition_means(wide, m, s, outdir / "figures" / f"fig_nps_condition_means_{m}_{s}.png")
            plot_direct_contrasts(direct, m, s, outdir / "figures" / f"fig_nps_direct_contrasts_{m}_{s}.png")
            plot_temperature_slopes(wide, m, s, outdir / "figures" / f"fig_nps_temperature_slopes_{m}_{s}.png")

    manifest = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "signature_values": str(sig_path),
        "signature_values_sha256": sha256_file(sig_path),
        "outdir": str(outdir),
        "signature": args.signature,
        "sets": sorted(df["set_name"].unique().tolist()),
        "measures": available_measures,
        "primary_set": args.primary_set,
        "primary_measure": args.primary_measure,
        "n_input_rows_ok_signature": int(df.shape[0]),
        "n_subjects_by_set": df.groupby("set_name")["subject"].nunique().to_dict(),
        "note": "Replication-style analysis using subject-level GLM signature expressions; not exact single-trial reproduction.",
    }
    save_json(manifest, outdir / "manifest.json")

    write_report(outdir, manifest, summary, direct, condition, temp, args.primary_set, args.primary_measure)

    print("Processing completed successfully.")
    print(f"Output directory: {outdir}")
    print("\nPrimary summary:")
    ps = summary[(summary["set_name"] == args.primary_set) & (summary["measure"] == args.primary_measure)]
    if not ps.empty:
        cols = [
            "set_name", "measure", "passive_temperature_mean", "passive_temperature_p",
            "up_minus_down_mean", "up_minus_down_p",
            "down_minus_passive_mean", "down_minus_passive_p",
            "original_like_pattern", "down_specific_divergence", "interpretation",
        ]
        print(ps[cols].to_string(index=False))
    else:
        print("No primary summary row found. Check --primary-set and --primary-measure.")


if __name__ == "__main__":
    main()
