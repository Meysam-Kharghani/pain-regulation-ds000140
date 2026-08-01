#!/usr/bin/env python3
"""
run_roi_signature_analysis.py

ROI and signature analysis for ds000140 using:
  - subject-level fixed-effect maps from group_level_analysis
  - ROI masks from roi_masks_analysis
  - signature maps from roi_masks_analysis
  - optional behavioral subject-level table from behavior_computational_analysis

Analyses
--------
1. Extract ROI mean beta values from subject fixed-effect maps.
2. Extract signature expression values from subject fixed-effect maps.
3. Run one-sample tests for each set/effect/ROI and set/effect/signature.
4. FDR-correct within each analysis family.
5. Compare robustness sets against main via subject-level correlations where possible.
6. Optionally test brain-behavior associations.

Analysis inputs
---------------
Main ROI analysis:
  --subject-effects group_level_analysis/tables/subject_effect_inventory.tsv
  --roi-definitions roi_masks_analysis/roi_definitions.tsv
  --signature-definitions roi_masks_analysis/signature_definitions.tsv

Small-ROI boundary:
  NAc/PAG estimates are treated as exploratory because smoothing sensitivity was not fully evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
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


def now_local_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def save_tsv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=False)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_subject(x) -> str:
    s = str(x)
    if s.startswith("sub-"):
        return s
    m = re.search(r"(\d+)", s)
    return f"sub-{int(m.group(1)):02d}" if m else s


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def cohen_dz(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    if sd < 1e-12:
        return np.nan
    return float(x.mean() / sd)


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 1) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = rng.choice(x, size=len(x), replace=True).mean()
    return tuple(np.percentile(vals, [2.5, 97.5]).astype(float))


def fdr_by_group(df: pd.DataFrame, p_col: str, group_cols: Sequence[str], out_col: str) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    if len(out) == 0:
        return out
    for _, idx in out.groupby(list(group_cols)).groups.items():
        idx = list(idx)
        p = pd.to_numeric(out.loc[idx, p_col], errors="coerce")
        mask = p.notna() & np.isfinite(p)
        if mask.sum() > 0:
            q = np.full(len(idx), np.nan)
            q[np.where(mask.values)[0]] = multipletests(p[mask].values, method="fdr_bh")[1]
            out.loc[idx, out_col] = q
    return out


def image_data(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img


def shape_key(img_or_data) -> Tuple[int, int, int]:
    if hasattr(img_or_data, "shape"):
        return tuple(img_or_data.shape[:3])
    return tuple(np.asarray(img_or_data).shape[:3])


# =============================================================================
# Loading inputs
# =============================================================================

def load_subject_effects(path: Path, sets: Optional[Sequence[str]] = None, effects: Optional[Sequence[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["subject"] = df["subject"].map(clean_subject)
    if sets:
        df = df[df["set_name"].isin(sets)].copy()
    if effects:
        df = df[df["effect"].isin(effects)].copy()
    df = df[df["beta_map"].astype(str) != ""].copy()
    df = df[df["beta_map"].map(lambda p: Path(str(p)).exists())].copy()
    return df.sort_values(["set_name", "effect", "subject"]).reset_index(drop=True)


def load_roi_definitions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df[df["status"].astype(str) == "OK"].copy()
    df = df[df["mask_path"].astype(str) != ""].copy()
    df = df[df["mask_path"].map(lambda p: Path(str(p)).exists())].copy()
    return df.reset_index(drop=True)


def load_signature_definitions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df[df["status"].astype(str) == "OK"].copy()
    df = df[df["resampled_map"].astype(str) != ""].copy()
    df = df[df["resampled_map"].map(lambda p: Path(str(p)).exists())].copy()
    return df.reset_index(drop=True)


def load_behavior(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    if "subject" not in df.columns:
        return pd.DataFrame()
    df["subject"] = df["subject"].map(clean_subject)
    return df


# =============================================================================
# Extraction
# =============================================================================

def extract_roi_values(subject_effects: pd.DataFrame, roi_defs: pd.DataFrame) -> pd.DataFrame:
    mask_cache: Dict[str, Tuple[np.ndarray, int]] = {}
    rows = []

    for _, roi in roi_defs.iterrows():
        rname = roi["roi"]
        mpath = Path(str(roi["mask_path"]))
        mdata, mimg = image_data(mpath)
        mask = mdata > 0.5
        mask_cache[rname] = (mask, int(mask.sum()))

    for _, eff in subject_effects.iterrows():
        beta_path = Path(str(eff["beta_map"]))
        data, img = image_data(beta_path)

        for _, roi in roi_defs.iterrows():
            rname = roi["roi"]
            mask, nvox = mask_cache[rname]

            if mask.shape != data.shape[:3]:
                rows.append({
                    "set_name": eff["set_name"],
                    "subject": eff["subject"],
                    "effect": eff["effect"],
                    "roi": rname,
                    "family": roi["family"],
                    "priority": roi.get("priority", ""),
                    "value": np.nan,
                    "n_voxels": nvox,
                    "status": "SHAPE_MISMATCH",
                    "map": str(beta_path),
                    "mask": str(roi["mask_path"]),
                })
                continue

            vals = data[mask]
            vals = vals[np.isfinite(vals)]
            value = float(vals.mean()) if len(vals) else np.nan
            rows.append({
                "set_name": eff["set_name"],
                "subject": eff["subject"],
                "effect": eff["effect"],
                "roi": rname,
                "family": roi["family"],
                "priority": roi.get("priority", ""),
                "value": value,
                "n_voxels": nvox,
                "status": "OK" if np.isfinite(value) else "EMPTY_OR_NAN",
                "map": str(beta_path),
                "mask": str(roi["mask_path"]),
            })

    return pd.DataFrame(rows)


def extract_signature_values(subject_effects: pd.DataFrame, sig_defs: pd.DataFrame) -> pd.DataFrame:
    sig_cache: Dict[str, Tuple[np.ndarray, np.ndarray, int, float]] = {}
    rows = []

    for _, sig in sig_defs.iterrows():
        sname = sig["signature"]
        wpath = Path(str(sig["resampled_map"]))
        wdata, wimg = image_data(wpath)
        weights = np.asarray(wdata, dtype=np.float32)
        mask = np.isfinite(weights) & (np.abs(weights) > 1e-12)
        norm_abs = float(np.sum(np.abs(weights[mask]))) if mask.sum() else np.nan
        norm_l2 = float(np.sqrt(np.sum(weights[mask] ** 2))) if mask.sum() else np.nan
        sig_cache[sname] = (weights, mask, int(mask.sum()), norm_abs, norm_l2)

    for _, eff in subject_effects.iterrows():
        beta_path = Path(str(eff["beta_map"]))
        data, img = image_data(beta_path)
        data = np.asarray(data, dtype=np.float32)

        for _, sig in sig_defs.iterrows():
            sname = sig["signature"]
            weights, mask, nvox, norm_abs, norm_l2 = sig_cache[sname]
            if weights.shape != data.shape[:3]:
                rows.append({
                    "set_name": eff["set_name"],
                    "subject": eff["subject"],
                    "effect": eff["effect"],
                    "signature": sname,
                    "family": "signature",
                    "signature_dot": np.nan,
                    "signature_weighted_mean": np.nan,
                    "signature_cosine": np.nan,
                    "n_voxels": nvox,
                    "status": "SHAPE_MISMATCH",
                    "map": str(beta_path),
                    "signature_map": str(sig["resampled_map"]),
                })
                continue

            x = data[mask]
            w = weights[mask]
            finite = np.isfinite(x) & np.isfinite(w)
            x = x[finite]
            w = w[finite]

            if len(x) == 0:
                dot = wmean = cosine = np.nan
                status = "EMPTY_OR_NAN"
            else:
                dot = float(np.sum(x * w))
                denom_abs = float(np.sum(np.abs(w)))
                wmean = float(dot / denom_abs) if denom_abs > 0 else np.nan
                denom_l2 = float(np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(w ** 2)))
                cosine = float(dot / denom_l2) if denom_l2 > 0 else np.nan
                status = "OK"

            rows.append({
                "set_name": eff["set_name"],
                "subject": eff["subject"],
                "effect": eff["effect"],
                "signature": sname,
                "family": "signature",
                "signature_dot": dot,
                "signature_weighted_mean": wmean,
                "signature_cosine": cosine,
                "n_voxels": nvox,
                "status": status,
                "map": str(beta_path),
                "signature_map": str(sig["resampled_map"]),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Statistics
# =============================================================================

def one_sample_stats(values: pd.Series, label_cols: Dict[str, str], value_name: str = "value", seed: int = 1) -> Dict[str, object]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    out = dict(label_cols)
    out["measure"] = value_name
    out["n"] = int(len(x))
    if len(x) < 2:
        out.update({
            "mean": np.nan, "sd": np.nan, "se": np.nan, "t": np.nan, "p": np.nan,
            "dz": np.nan, "ci_lo_boot": np.nan, "ci_hi_boot": np.nan,
        })
        return out

    tt = stats.ttest_1samp(x, 0.0)
    lo, hi = bootstrap_mean_ci(x, n_boot=5000, seed=seed)
    out.update({
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "se": float(x.std(ddof=1) / np.sqrt(len(x))),
        "t": float(tt.statistic),
        "p": float(tt.pvalue),
        "dz": cohen_dz(x),
        "ci_lo_boot": lo,
        "ci_hi_boot": hi,
    })
    return out


def roi_stats(roi_values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = roi_values[roi_values["status"] == "OK"].copy()
    for (set_name, effect, family, roi), g in d.groupby(["set_name", "effect", "family", "roi"]):
        rows.append(one_sample_stats(
            g["value"],
            {"set_name": set_name, "effect": effect, "family": family, "roi": roi},
            value_name="roi_mean_beta",
            seed=stable_seed(set_name, effect, roi),
        ))
    out = pd.DataFrame(rows)
    if len(out):
        out = fdr_by_group(out, p_col="p", group_cols=["set_name", "effect", "family"], out_col="q_fdr_family")
        out = fdr_by_group(out, p_col="p", group_cols=["set_name", "effect"], out_col="q_fdr_effect_all_rois")
    return out


def signature_stats(sig_values: pd.DataFrame, measure: str = "signature_weighted_mean") -> pd.DataFrame:
    rows = []
    d = sig_values[sig_values["status"] == "OK"].copy()
    for (set_name, effect, sig), g in d.groupby(["set_name", "effect", "signature"]):
        rows.append(one_sample_stats(
            g[measure],
            {"set_name": set_name, "effect": effect, "family": "signature", "signature": sig},
            value_name=measure,
            seed=stable_seed(set_name, effect, sig),
        ))
    out = pd.DataFrame(rows)
    if len(out):
        out = fdr_by_group(out, p_col="p", group_cols=["set_name", "effect"], out_col="q_fdr_signature_family")
    return out


def roi_robustness(roi_values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = roi_values[roi_values["status"] == "OK"].copy()
    main = d[d["set_name"] == "main"].copy()
    if main.empty:
        return pd.DataFrame(rows)

    for (effect, roi), gm in main.groupby(["effect", "roi"]):
        for comp_set, gc in d[(d["effect"] == effect) & (d["roi"] == roi) & (d["set_name"] != "main")].groupby("set_name"):
            mm = gm[["subject", "value"]].rename(columns={"value": "main_value"})
            cc = gc[["subject", "value"]].rename(columns={"value": "comparison_value"})
            merged = mm.merge(cc, on="subject", how="inner")
            r = np.nan
            p = np.nan
            if len(merged) >= 3 and merged["main_value"].std(ddof=0) > 0 and merged["comparison_value"].std(ddof=0) > 0:
                r, p = stats.pearsonr(merged["main_value"], merged["comparison_value"])
            rows.append({
                "effect": effect,
                "roi": roi,
                "comparison_set": comp_set,
                "n_overlap_subjects": int(len(merged)),
                "pearson_r": float(r) if np.isfinite(r) else np.nan,
                "pearson_p": float(p) if np.isfinite(p) else np.nan,
                "main_mean": float(gm["value"].mean()),
                "comparison_mean": float(gc["value"].mean()),
            })
    return pd.DataFrame(rows)


def behavior_associations(roi_values: pd.DataFrame, sig_values: pd.DataFrame, behavior: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if behavior.empty:
        return pd.DataFrame(), pd.DataFrame()

    candidate_cols = [
        "mean_reg_success_z",
        "reg_success_z_up",
        "reg_success_z_down",
        "success_asymmetry_z_up_minus_down",
        "task_separation_rating_z_up_minus_down",
        "passive_sensory_slope_z",
        "passive_withinrun_slope_z",
        "passive_acrossrun_slope_z",
    ]
    bcols = [c for c in candidate_cols if c in behavior.columns]
    if not bcols:
        return pd.DataFrame(), pd.DataFrame()

    # Use the primary analysis set by default to limit the exploratory output size.
    rv = roi_values[(roi_values["set_name"] == "main") & (roi_values["status"] == "OK")].copy()
    sv = sig_values[(sig_values["set_name"] == "main") & (sig_values["status"] == "OK")].copy()

    roi_rows = []
    for (effect, roi, family), g in rv.groupby(["effect", "roi", "family"]):
        dat = g[["subject", "value"]].merge(behavior[["subject"] + bcols], on="subject", how="inner")
        for bc in bcols:
            x = pd.to_numeric(dat["value"], errors="coerce")
            y = pd.to_numeric(dat[bc], errors="coerce")
            m = x.notna() & y.notna()
            r = p = rho = ps = np.nan
            if m.sum() >= 5 and x[m].std(ddof=0) > 0 and y[m].std(ddof=0) > 0:
                r, p = stats.pearsonr(x[m], y[m])
                rho, ps = stats.spearmanr(x[m], y[m])
            roi_rows.append({
                "effect": effect, "roi": roi, "family": family,
                "behavior": bc, "n": int(m.sum()),
                "pearson_r": r, "pearson_p": p,
                "spearman_rho": rho, "spearman_p": ps,
            })

    sig_rows = []
    for (effect, sig), g in sv.groupby(["effect", "signature"]):
        dat = g[["subject", "signature_weighted_mean"]].merge(behavior[["subject"] + bcols], on="subject", how="inner")
        for bc in bcols:
            x = pd.to_numeric(dat["signature_weighted_mean"], errors="coerce")
            y = pd.to_numeric(dat[bc], errors="coerce")
            m = x.notna() & y.notna()
            r = p = rho = ps = np.nan
            if m.sum() >= 5 and x[m].std(ddof=0) > 0 and y[m].std(ddof=0) > 0:
                r, p = stats.pearsonr(x[m], y[m])
                rho, ps = stats.spearmanr(x[m], y[m])
            sig_rows.append({
                "effect": effect, "signature": sig, "family": "signature",
                "behavior": bc, "n": int(m.sum()),
                "pearson_r": r, "pearson_p": p,
                "spearman_rho": rho, "spearman_p": ps,
            })

    roi_assoc = pd.DataFrame(roi_rows)
    sig_assoc = pd.DataFrame(sig_rows)

    if len(roi_assoc):
        roi_assoc = fdr_by_group(roi_assoc, p_col="pearson_p", group_cols=["effect", "family"], out_col="pearson_q_fdr")
        roi_assoc = fdr_by_group(roi_assoc, p_col="spearman_p", group_cols=["effect", "family"], out_col="spearman_q_fdr")
    if len(sig_assoc):
        sig_assoc = fdr_by_group(sig_assoc, p_col="pearson_p", group_cols=["effect"], out_col="pearson_q_fdr")
        sig_assoc = fdr_by_group(sig_assoc, p_col="spearman_p", group_cols=["effect"], out_col="spearman_q_fdr")

    return roi_assoc, sig_assoc


# =============================================================================
# FDR
# =============================================================================

def fdr_by_group(df: pd.DataFrame, p_col: str, group_cols: Sequence[str], out_col: str) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    if len(out) == 0 or p_col not in out.columns:
        return out

    for _, idx in out.groupby(list(group_cols)).groups.items():
        idx = list(idx)
        p = pd.to_numeric(out.loc[idx, p_col], errors="coerce")
        m = p.notna() & np.isfinite(p)
        if m.sum() > 0:
            q = np.full(len(idx), np.nan)
            q[np.where(m.values)[0]] = multipletests(p[m].values, method="fdr_bh")[1]
            out.loc[idx, out_col] = q
    return out


# =============================================================================
# Report
# =============================================================================

def write_report(outdir: Path, manifest: Dict, roi_values: pd.DataFrame, roi_stats_df: pd.DataFrame, sig_stats_df: pd.DataFrame, roi_assoc: pd.DataFrame, sig_assoc: pd.DataFrame) -> None:
    lines = []
    lines.append("# ROI and signature analysis report\n\n")
    lines.append("## Overview\n")
    lines.append(f"- Generated: {manifest['timestamp_local']}\n")
    lines.append(f"- Subject effects: `{manifest['subject_effects']}`\n")
    lines.append(f"- ROI definitions: `{manifest['roi_definitions']}`\n")
    lines.append(f"- Signature definitions: `{manifest['signature_definitions']}`\n")
    lines.append(f"- Output directory: `{manifest['outdir']}`\n\n")

    lines.append("## Extraction summary\n")
    lines.append(f"- ROI value rows: {len(roi_values)}\n")
    if len(roi_values):
        lines.append(f"- ROI extraction OK rows: {int((roi_values['status'] == 'OK').sum())}\n")
        lines.append(f"- ROI count: {roi_values['roi'].nunique()}\n")
        lines.append(f"- Set count: {roi_values['set_name'].nunique()}\n")
        lines.append(f"- Effect count: {roi_values['effect'].nunique()}\n")
    lines.append(f"- ROI stats rows: {len(roi_stats_df)}\n")
    lines.append(f"- Signature stats rows: {len(sig_stats_df)}\n\n")

    lines.append("## Strongest ROI effects by p-value\n")
    if len(roi_stats_df):
        top = roi_stats_df.sort_values("p").head(20)
        for _, r in top.iterrows():
            lines.append(
                f"- {r['set_name']} | {r['effect']} | {r['roi']} | "
                f"mean={r['mean']:.4g}, t={r['t']:.3f}, p={r['p']:.3g}, q_family={r.get('q_fdr_family', np.nan):.3g}\n"
            )
    lines.append("\n")

    lines.append("## Strongest signature effects by p-value\n")
    if len(sig_stats_df):
        top = sig_stats_df.sort_values("p").head(20)
        for _, r in top.iterrows():
            lines.append(
                f"- {r['set_name']} | {r['effect']} | {r['signature']} | "
                f"mean={r['mean']:.4g}, t={r['t']:.3f}, p={r['p']:.3g}, q={r.get('q_fdr_signature_family', np.nan):.3g}\n"
            )
    lines.append("\n")

    lines.append("## Brain-behavior associations\n")
    if len(roi_assoc):
        sig = roi_assoc.sort_values("pearson_p").head(15)
        lines.append("### Top ROI associations\n")
        for _, r in sig.iterrows():
            lines.append(
                f"- {r['effect']} | {r['roi']} | {r['behavior']}: "
                f"r={r['pearson_r']:.3f}, p={r['pearson_p']:.3g}, q={r.get('pearson_q_fdr', np.nan):.3g}\n"
            )
    if len(sig_assoc):
        sig = sig_assoc.sort_values("pearson_p").head(15)
        lines.append("### Top signature associations\n")
        for _, r in sig.iterrows():
            lines.append(
                f"- {r['effect']} | {r['signature']} | {r['behavior']}: "
                f"r={r['pearson_r']:.3f}, p={r['pearson_p']:.3g}, q={r.get('pearson_q_fdr', np.nan):.3g}\n"
            )
    if not len(roi_assoc) and not len(sig_assoc):
        lines.append("- No behavior table provided or no matching behavior columns found.\n")
    lines.append("\n")

    lines.append("## Interpretation cautions\n")
    lines.append("- ROI-family inference uses FDR correction within each family/effect.\n")
    lines.append("- NAc and PAG are small ROIs; their estimates are treated as exploratory because smoothing sensitivity was not fully evaluated.\n")
    lines.append("- NPS was not present in the available signature bundle, whereas SIIPS1 and other CANlab signatures were available.\n")
    lines.append("- Brain-behavior associations are exploratory because they were not preregistered and the sample size is approximately 33.\n")

    (outdir / "roi_signature_analysis_report.md").write_text("".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-effects", required=True)
    ap.add_argument("--roi-definitions", required=True)
    ap.add_argument("--signature-definitions", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--behavior-subjects", default="")
    ap.add_argument("--sets", default="", help="Optional comma-separated sets to include")
    ap.add_argument("--effects", default="", help="Optional comma-separated effects to include")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)
    tables = outdir / "tables"
    ensure_dir(tables)

    sets = [s.strip() for s in args.sets.split(",") if s.strip()] if args.sets else None
    effects = [e.strip() for e in args.effects.split(",") if e.strip()] if args.effects else None

    subject_effects = load_subject_effects(Path(args.subject_effects).expanduser().resolve(), sets=sets, effects=effects)
    roi_defs = load_roi_definitions(Path(args.roi_definitions).expanduser().resolve())
    sig_defs = load_signature_definitions(Path(args.signature_definitions).expanduser().resolve())
    behavior = load_behavior(Path(args.behavior_subjects).expanduser().resolve()) if args.behavior_subjects else pd.DataFrame()

    roi_values = extract_roi_values(subject_effects, roi_defs)
    sig_values = extract_signature_values(subject_effects, sig_defs)

    roi_stats_df = roi_stats(roi_values)
    sig_stats_df = signature_stats(sig_values, measure="signature_weighted_mean")
    robust = roi_robustness(roi_values)

    roi_assoc, sig_assoc = behavior_associations(roi_values, sig_values, behavior)

    save_tsv(roi_values, tables / "roi_values.tsv")
    save_tsv(sig_values, tables / "signature_values.tsv")
    save_tsv(roi_stats_df, tables / "roi_stats.tsv")
    save_tsv(sig_stats_df, tables / "signature_stats.tsv")
    save_tsv(robust, tables / "roi_robustness.tsv")
    save_tsv(roi_assoc, tables / "roi_behavior_associations.tsv")
    save_tsv(sig_assoc, tables / "signature_behavior_associations.tsv")

    manifest = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "python_version": sys.version.replace("\n", " "),
        "platform": os.uname().sysname if hasattr(os, "uname") else "",
        "subject_effects": str(Path(args.subject_effects).expanduser().resolve()),
        "roi_definitions": str(Path(args.roi_definitions).expanduser().resolve()),
        "signature_definitions": str(Path(args.signature_definitions).expanduser().resolve()),
        "behavior_subjects": str(Path(args.behavior_subjects).expanduser().resolve()) if args.behavior_subjects else "",
        "outdir": str(outdir),
        "sets": sets or "all",
        "effects": effects or "all",
        "n_subject_effect_rows": int(len(subject_effects)),
        "n_roi_defs": int(len(roi_defs)),
        "n_signature_defs": int(len(sig_defs)),
        "n_roi_value_rows": int(len(roi_values)),
        "n_signature_value_rows": int(len(sig_values)),
    }
    save_json(manifest, outdir / "manifest.json")
    write_report(outdir, manifest, roi_values, roi_stats_df, sig_stats_df, roi_assoc, sig_assoc)

    print("Processing completed successfully.")
    print(f"Subject effect rows: {len(subject_effects)}")
    print(f"ROI definitions OK: {len(roi_defs)}")
    print(f"Signature definitions OK: {len(sig_defs)}")
    print(f"ROI value rows: {len(roi_values)}")
    print(f"Signature value rows: {len(sig_values)}")
    print(f"Output: {outdir}")


if __name__ == "__main__":
    import argparse
    main()
