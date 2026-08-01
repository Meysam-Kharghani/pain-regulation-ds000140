#!/usr/bin/env python3
"""Targeted ROI, connectivity, and brain-behavior pathway screening.

The workflow restricts ROI activity, beta-series functional connectivity, and
lagged predictive-connectivity tests to prespecified regulatory, value,
interoceptive, sensory, and descending-modulation pathways. It constructs
participant-level pathway scores and evaluates associations with behavioral and
computational variables.

Lagged predictive-connectivity estimates are exploratory regression summaries;
they are not Dynamic Causal Modeling estimates and do not establish causal or
physiological directionality.

Required inputs are the participant-level ROI activity summary, beta-series
functional-connectivity edges, and lagged predictive-connectivity edges. A
participant-level behavioral table may be supplied for brain-behavior tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_tsv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=False)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def clean_subject(x) -> str:
    s = str(x)
    if s.startswith("sub-"):
        return s
    m = re.search(r"(\d+)", s)
    return f"sub-{int(m.group(1)):02d}" if m else s


def cohen_dz(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 1e-12 else np.nan


def one_sample_test(values, label: str) -> Dict[str, object]:
    x = safe_num(pd.Series(values)).dropna().to_numpy(float)
    out = {"quantity": label, "n": int(len(x))}
    if len(x) < 2:
        out.update({"mean": np.nan, "sd": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "dz": np.nan})
        return out
    tt = stats.ttest_1samp(x, 0.0)
    sd = x.std(ddof=1)
    out.update({
        "mean": float(x.mean()),
        "sd": float(sd),
        "se": float(sd / np.sqrt(len(x))),
        "t": float(tt.statistic),
        "p": float(tt.pvalue),
        "dz": cohen_dz(x),
    })
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(x).pvalue)
    except Exception:
        out["wilcoxon_p"] = np.nan
    return out


def corr_test(x, y, label: str = "") -> Dict[str, object]:
    d = pd.DataFrame({
        "x": safe_num(pd.Series(x)),
        "y": safe_num(pd.Series(y)),
    }).dropna()
    out = {"comparison": label, "n": int(len(d))}
    if len(d) < 5 or d["x"].std(ddof=1) < 1e-12 or d["y"].std(ddof=1) < 1e-12:
        out.update({"r": np.nan, "p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan})
        return out
    pr = stats.pearsonr(d["x"], d["y"])
    sr = stats.spearmanr(d["x"], d["y"])
    out.update({
        "r": float(pr.statistic),
        "p": float(pr.pvalue),
        "spearman_rho": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    })
    return out


def fdr(df: pd.DataFrame, p_col: str, q_col: str, by: Sequence[str] = ()) -> pd.DataFrame:
    out = df.copy()
    out[q_col] = np.nan
    if out.empty or p_col not in out.columns:
        return out

    groups = out.groupby(list(by), dropna=False).groups.values() if by else [out.index]
    for idx in groups:
        p = pd.to_numeric(out.loc[idx, p_col], errors="coerce")
        mask = p.notna() & np.isfinite(p)
        if mask.sum() > 0:
            out.loc[p[mask].index, q_col] = multipletests(p[mask], method="fdr_bh")[1]
    return out


def contains_any(text: str, patterns: Sequence[str]) -> bool:
    text = str(text)
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def edge_matches(a: str, b: str, pat_a: Sequence[str], pat_b: Sequence[str]) -> bool:
    return (contains_any(a, pat_a) and contains_any(b, pat_b)) or (contains_any(a, pat_b) and contains_any(b, pat_a))


def directed_matches(src: str, tgt: str, src_pat: Sequence[str], tgt_pat: Sequence[str]) -> bool:
    return contains_any(src, src_pat) and contains_any(tgt, tgt_pat)


# =============================================================================
# Target definitions
# =============================================================================

REG_QUANTITIES = ["up_minus_down", "up_minus_passive", "down_minus_passive", "regulation_mean_minus_passive"]
STATE_QUANTITIES_FC = ["z_passive", "z_up", "z_down"]
STATE_QUANTITIES_EC = ["ec_passive", "ec_up", "ec_down"]
STATE_QUANTITIES_ROI = ["beta_passive", "beta_up", "beta_down"]

# Main hypothesis-driven ROI targets
ROI_TARGETS = {
    "control_dlPFC_IFJ_L": [r"dlPFC_IFJ_L", r"dlPFC.*L", r"IFJ.*L"],
    "control_dlPFC_IFJ_R": [r"dlPFC_IFJ_R", r"dlPFC.*R", r"IFJ.*R"],
    "control_SMA_preSMA": [r"SMA_preSMA", r"preSMA", r"SMA"],
    "value_vmPFC_mOFC": [r"vmPFC", r"mOFC", r"orbitofrontal"],
    "value_NAc": [r"NAc", r"accumbens"],
    "descending_pgACC_rACC": [r"pgACC", r"rACC", r"perigenual", r"rostral"],
    "descending_PAG": [r"PAG", r"periaqueductal"],
    "salience_aINS": [r"anterior_insula", r"aINS"],
    "salience_dACC_MCC": [r"dACC", r"MCC", r"cingulate"],
    "sensory_S1_S2_thalamus": [r"S1", r"S2", r"operculum", r"thalamus"],
}

# Undirected FC targets
FC_TARGETS = [
    {
        "pathway": "NAc_vmPFC_value",
        "family": "value_regulation",
        "a_patterns": [r"NAc", r"accumbens"],
        "b_patterns": [r"vmPFC", r"mOFC", r"orbitofrontal"],
    },
    {
        "pathway": "dlPFC_IFJ_L_vmPFC_control_value",
        "family": "control_value",
        "a_patterns": [r"dlPFC_IFJ_L", r"dlPFC.*L", r"IFJ.*L"],
        "b_patterns": [r"vmPFC", r"mOFC", r"orbitofrontal"],
    },
    {
        "pathway": "SMA_preSMA_dlPFC_IFJ_control",
        "family": "control_implementation",
        "a_patterns": [r"SMA_preSMA", r"preSMA", r"SMA"],
        "b_patterns": [r"dlPFC", r"IFJ"],
    },
    {
        "pathway": "pgACC_rACC_PAG_descending",
        "family": "descending_modulation",
        "a_patterns": [r"pgACC", r"rACC", r"perigenual", r"rostral"],
        "b_patterns": [r"PAG", r"periaqueductal"],
    },
    {
        "pathway": "aINS_dACC_MCC_salience",
        "family": "salience_interoception",
        "a_patterns": [r"anterior_insula", r"aINS", r"insula"],
        "b_patterns": [r"dACC", r"MCC", r"cingulate"],
    },
    {
        "pathway": "sensory_thalamus_S1_S2_insula",
        "family": "sensory_interoceptive",
        "a_patterns": [r"thalamus"],
        "b_patterns": [r"S1", r"S2", r"operculum", r"insula"],
    },
]

# Directed EC-lite targets
EC_TARGETS = [
    {
        "pathway": "NAc_to_vmPFC_value",
        "family": "value_regulation",
        "src_patterns": [r"NAc", r"accumbens"],
        "tgt_patterns": [r"vmPFC", r"mOFC", r"orbitofrontal"],
    },
    {
        "pathway": "vmPFC_to_NAc_value",
        "family": "value_regulation",
        "src_patterns": [r"vmPFC", r"mOFC", r"orbitofrontal"],
        "tgt_patterns": [r"NAc", r"accumbens"],
    },
    {
        "pathway": "dlPFC_IFJ_L_to_vmPFC_control_value",
        "family": "control_value",
        "src_patterns": [r"dlPFC_IFJ_L", r"dlPFC.*L", r"IFJ.*L"],
        "tgt_patterns": [r"vmPFC", r"mOFC", r"orbitofrontal"],
    },
    {
        "pathway": "vmPFC_to_dlPFC_IFJ_L_feedback",
        "family": "control_value",
        "src_patterns": [r"vmPFC", r"mOFC", r"orbitofrontal"],
        "tgt_patterns": [r"dlPFC_IFJ_L", r"dlPFC.*L", r"IFJ.*L"],
    },
    {
        "pathway": "SMA_preSMA_to_dlPFC_IFJ_control",
        "family": "control_implementation",
        "src_patterns": [r"SMA_preSMA", r"preSMA", r"SMA"],
        "tgt_patterns": [r"dlPFC", r"IFJ"],
    },
    {
        "pathway": "dlPFC_IFJ_to_SMA_preSMA_control",
        "family": "control_implementation",
        "src_patterns": [r"dlPFC", r"IFJ"],
        "tgt_patterns": [r"SMA_preSMA", r"preSMA", r"SMA"],
    },
    {
        "pathway": "pgACC_rACC_to_PAG_descending",
        "family": "descending_modulation",
        "src_patterns": [r"pgACC", r"rACC", r"perigenual", r"rostral"],
        "tgt_patterns": [r"PAG", r"periaqueductal"],
    },
    {
        "pathway": "PAG_to_pgACC_rACC_feedback",
        "family": "descending_modulation",
        "src_patterns": [r"PAG", r"periaqueductal"],
        "tgt_patterns": [r"pgACC", r"rACC", r"perigenual", r"rostral"],
    },
    {
        "pathway": "aINS_to_dACC_MCC_salience",
        "family": "salience_interoception",
        "src_patterns": [r"anterior_insula", r"aINS", r"insula"],
        "tgt_patterns": [r"dACC", r"MCC", r"cingulate"],
    },
    {
        "pathway": "dACC_MCC_to_aINS_salience",
        "family": "salience_interoception",
        "src_patterns": [r"dACC", r"MCC", r"cingulate"],
        "tgt_patterns": [r"anterior_insula", r"aINS", r"insula"],
    },
]


# =============================================================================
# Loading and targeting
# =============================================================================

def load_behavior(path: Optional[Path], behavior_regex: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    beh = pd.read_csv(path, sep="\t")
    if "subject" not in beh.columns:
        raise ValueError("Behavior file must contain subject column.")
    beh = beh.copy()
    beh["subject"] = beh["subject"].map(clean_subject)

    # Numeric behavior columns of interest.
    numeric_cols = []
    for c in beh.columns:
        if c == "subject":
            continue
        vals = safe_num(beh[c])
        if vals.notna().sum() >= 5 and vals.std(skipna=True) > 1e-12:
            numeric_cols.append(c)
            beh[c] = vals

    if behavior_regex:
        pat = re.compile(behavior_regex, flags=re.IGNORECASE)
        numeric_cols = [c for c in numeric_cols if pat.search(c)]

    keep = ["subject"] + numeric_cols
    return beh[keep].drop_duplicates("subject")


def target_roi_activity(roi_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_name, pats in ROI_TARGETS.items():
        d = roi_summary[roi_summary["roi"].map(lambda x: contains_any(x, pats))].copy()
        if d.empty:
            continue
        d["target"] = target_name
        d["family"] = target_name.split("_")[0]
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["target", "subject", "roi", "roi_safe"])
    return out


def target_fc(fc_edges: pd.DataFrame) -> pd.DataFrame:
    if fc_edges.empty:
        return pd.DataFrame()
    rows = []
    for spec in FC_TARGETS:
        d = fc_edges[
            fc_edges.apply(
                lambda r: edge_matches(str(r.get("roi_a", "")), str(r.get("roi_b", "")), spec["a_patterns"], spec["b_patterns"]),
                axis=1,
            )
        ].copy()
        if d.empty:
            continue
        d["target_pathway"] = spec["pathway"]
        d["target_family"] = spec["family"]
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(["target_pathway", "subject", "roi_a", "roi_b", "quantity"])


def target_ec(ec_edges: pd.DataFrame) -> pd.DataFrame:
    if ec_edges.empty:
        return pd.DataFrame()
    rows = []
    for spec in EC_TARGETS:
        d = ec_edges[
            ec_edges.apply(
                lambda r: directed_matches(str(r.get("source_roi", "")), str(r.get("target_roi", "")), spec["src_patterns"], spec["tgt_patterns"]),
                axis=1,
            )
        ].copy()
        if d.empty:
            continue
        d["target_pathway"] = spec["pathway"]
        d["target_family"] = spec["family"]
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(["target_pathway", "subject", "source_roi", "target_roi", "quantity"])


# =============================================================================
# Tests
# =============================================================================

def test_roi_targets(roi_targ: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if roi_targ.empty:
        return pd.DataFrame()
    quantities = STATE_QUANTITIES_ROI + REG_QUANTITIES
    for (target, family, roi, roi_safe), d in roi_targ.groupby(["target", "family", "roi", "roi_safe"]):
        for q in quantities:
            if q not in d.columns:
                continue
            row = one_sample_test(d[q], q)
            row.update({
                "modality": "ROI_activity",
                "target": target,
                "family": family,
                "roi": roi,
                "roi_safe": roi_safe,
            })
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = fdr(out, "p", "q_fdr_targeted_roi", by=["quantity"])
    return out


def test_fc_targets(fc_targ: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if fc_targ.empty:
        return pd.DataFrame()
    for (pathway, family, a, b, q), d in fc_targ.groupby(["target_pathway", "target_family", "roi_a", "roi_b", "quantity"]):
        row = one_sample_test(d["z_value"], q)
        row.update({
            "modality": "FC_beta_series",
            "target_pathway": pathway,
            "family": family,
            "roi_a": a,
            "roi_b": b,
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = fdr(out, "p", "q_fdr_targeted_fc", by=["quantity"])
    return out


def test_ec_targets(ec_targ: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if ec_targ.empty:
        return pd.DataFrame()
    for (pathway, family, src, tgt, q), d in ec_targ.groupby(["target_pathway", "target_family", "source_roi", "target_roi", "quantity"]):
        row = one_sample_test(d["ec_beta"], q)
        row.update({
            "modality": "EC_lite_directed_lagged",
            "target_pathway": pathway,
            "family": family,
            "source_roi": src,
            "target_roi": tgt,
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = fdr(out, "p", "q_fdr_targeted_ec", by=["quantity"])
    return out


def build_pathway_scores(roi_targ: pd.DataFrame, fc_targ: pd.DataFrame, ec_targ: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # ROI target composites: mean across matching ROIs in the target.
    if not roi_targ.empty:
        quantities = [q for q in STATE_QUANTITIES_ROI + REG_QUANTITIES if q in roi_targ.columns]
        for q in quantities:
            g = roi_targ.groupby(["subject", "target", "family"], as_index=False)[q].mean()
            g = g.rename(columns={q: "score"})
            g["modality"] = "ROI_activity"
            g["quantity"] = q
            rows.append(g)

    # FC target composites: mean across matched edges.
    if not fc_targ.empty:
        g = fc_targ.groupby(["subject", "target_pathway", "target_family", "quantity"], as_index=False)["z_value"].mean()
        g = g.rename(columns={
            "target_pathway": "target",
            "target_family": "family",
            "z_value": "score",
        })
        g["modality"] = "FC_beta_series"
        rows.append(g)

    # EC target composites: mean across matched directed edges.
    if not ec_targ.empty:
        g = ec_targ.groupby(["subject", "target_pathway", "target_family", "quantity"], as_index=False)["ec_beta"].mean()
        g = g.rename(columns={
            "target_pathway": "target",
            "target_family": "family",
            "ec_beta": "score",
        })
        g["modality"] = "EC_lite_directed_lagged"
        rows.append(g)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[["subject", "modality", "target", "family", "quantity", "score"]]


def test_pathway_scores(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if scores.empty:
        return pd.DataFrame()
    for (modality, target, family, q), d in scores.groupby(["modality", "target", "family", "quantity"]):
        row = one_sample_test(d["score"], q)
        row.update({"modality": modality, "target": target, "family": family})
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = fdr(out, "p", "q_fdr_pathway_scores", by=["modality", "quantity"])
    return out


def behavior_correlations(scores: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    if scores.empty or behavior.empty:
        return pd.DataFrame()
    d = scores.merge(behavior, on="subject", how="inner")
    beh_cols = [c for c in behavior.columns if c != "subject"]
    rows = []
    for (modality, target, family, q), dd in d.groupby(["modality", "target", "family", "quantity"]):
        for bc in beh_cols:
            row = corr_test(dd["score"], dd[bc], label=f"{modality}:{target}:{q}~{bc}")
            row.update({
                "modality": modality,
                "target": target,
                "family": family,
                "quantity": q,
                "behavior_variable": bc,
            })
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = fdr(out, "p", "q_fdr_behavior_corr", by=["modality", "quantity"])
    return out


def roi_behavior_correlations(roi_targ: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    if roi_targ.empty or behavior.empty:
        return pd.DataFrame()
    rows = []
    beh_cols = [c for c in behavior.columns if c != "subject"]
    quantities = [q for q in STATE_QUANTITIES_ROI + REG_QUANTITIES if q in roi_targ.columns]
    d = roi_targ.merge(behavior, on="subject", how="inner")
    for (target, family, roi, roi_safe), dd in d.groupby(["target", "family", "roi", "roi_safe"]):
        for q in quantities:
            for bc in beh_cols:
                row = corr_test(dd[q], dd[bc], label=f"{roi}:{q}~{bc}")
                row.update({
                    "target": target,
                    "family": family,
                    "roi": roi,
                    "roi_safe": roi_safe,
                    "quantity": q,
                    "behavior_variable": bc,
                })
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = fdr(out, "p", "q_fdr_roi_behavior_corr", by=["quantity"])
    return out


def family_summary(pathway_tests: pd.DataFrame, roi_tests: pd.DataFrame, fc_tests: pd.DataFrame, ec_tests: pd.DataFrame, behavior_corr: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_minp(df, label, qcol):
        if df.empty:
            return
        d = df.copy()
        if "family" not in d.columns or qcol not in d.columns:
            return
        for fam, dd in d.groupby("family"):
            valid = dd[pd.to_numeric(dd[qcol], errors="coerce").notna()].copy()
            if valid.empty:
                continue
            sort_cols = [qcol] + (["p"] if "p" in valid.columns else [])
            tie_cols = [c for c in ["quantity", "target", "target_pathway", "roi"] if c in valid.columns]
            best = valid.sort_values(sort_cols + tie_cols, kind="mergesort").iloc[0].to_dict()
            rows.append({
                "family": fam,
                "source_table": label,
                "best_q": best.get(qcol, np.nan),
                "best_p": best.get("p", np.nan),
                "best_quantity": best.get("quantity", ""),
                "best_target": best.get("target", best.get("target_pathway", "")),
                "best_description": best.get("roi", best.get("target", best.get("target_pathway", ""))),
            })

    add_minp(pathway_tests, "pathway_score_tests", "q_fdr_pathway_scores")
    add_minp(roi_tests, "targeted_roi_activity_tests", "q_fdr_targeted_roi")
    add_minp(fc_tests, "targeted_fc_tests", "q_fdr_targeted_fc")
    add_minp(ec_tests, "targeted_ec_tests", "q_fdr_targeted_ec")
    if not behavior_corr.empty:
        d = behavior_corr.copy()
        if "family" in d.columns and "q_fdr_behavior_corr" in d.columns:
            for fam, dd in d.groupby("family"):
                valid = dd[pd.to_numeric(dd["q_fdr_behavior_corr"], errors="coerce").notna()].copy()
                if valid.empty:
                    continue
                sort_cols = ["q_fdr_behavior_corr", "p", "behavior_variable", "target", "quantity"]
                best = valid.sort_values(sort_cols, kind="mergesort").iloc[0].to_dict()
                rows.append({
                    "family": fam,
                    "source_table": "pathway_behavior_correlations",
                    "best_q": best.get("q_fdr_behavior_corr", np.nan),
                    "best_p": best.get("p", np.nan),
                    "best_quantity": best.get("quantity", ""),
                    "best_target": best.get("target", ""),
                    "best_description": f"{best.get('target','')} ~ {best.get('behavior_variable','')}",
                })

    return pd.DataFrame(rows)


def portable_path(path: Path | None, base: Path) -> str:
    """Return a repository-relative or basename-only path for metadata."""
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.name


# =============================================================================
# Report
# =============================================================================

def write_report(outdir: Path,
                 manifest: Dict,
                 roi_tests: pd.DataFrame,
                 fc_tests: pd.DataFrame,
                 ec_tests: pd.DataFrame,
                 score_tests: pd.DataFrame,
                 behavior_corr: pd.DataFrame,
                 family_sum: pd.DataFrame) -> None:
    lines = []
    lines.append("# Targeted pathway screening report\n\n")
    lines.append("## Purpose\n")
    lines.append(
        "This analysis evaluates a prespecified set of regulatory, value, "
        "salience/interoceptive, sensory, and descending-modulation pathways. It combines targeted ROI activity, "
        "beta-series functional connectivity, lagged predictive connectivity, and brain-behavior associations.\n\n"
    )
    lines.append("## Inputs\n")
    lines.append(f"- ROI activity summary: `{manifest['roi_activity_summary']}`\n")
    lines.append(f"- FC edges: `{manifest['fc_edges']}`\n")
    lines.append(f"- EC edges: `{manifest['ec_edges']}`\n")
    lines.append(f"- Behavior subjects: `{manifest['behavior_subjects']}`\n")
    lines.append(f"- Output: `{manifest['outdir']}`\n\n")

    lines.append("## Interpretation of lagged predictive connectivity\n")
    lines.append(
        "Lagged predictive-connectivity results are regression-based temporal association estimates, not DCM. "
        "They are interpreted as exploratory pathway-prioritization measures and do not establish causal direction.\n\n"
    )

    def top_section(title, df, qcol, desc_cols):
        lines.append(f"## {title}\n")
        if df.empty or qcol not in df.columns:
            lines.append("No results available.\n\n")
            return
        d = df[pd.to_numeric(df[qcol], errors="coerce").notna()].sort_values(qcol).head(20)
        if d.empty:
            lines.append("No valid corrected p-values available.\n\n")
            return
        for _, r in d.iterrows():
            desc = " | ".join(str(r.get(c, "")) for c in desc_cols if c in r.index)
            lines.append(
                f"- {desc}; quantity={r.get('quantity','')}; mean/r={r.get('mean', r.get('r', np.nan)):.4g}; "
                f"p={r.get('p', np.nan):.4g}; q={r.get(qcol, np.nan):.4g}; n={r.get('n', '')}\n"
            )
        lines.append("\n")

    top_section("Top targeted ROI activity tests", roi_tests, "q_fdr_targeted_roi", ["target", "roi"])
    top_section("Top targeted FC tests", fc_tests, "q_fdr_targeted_fc", ["target_pathway", "roi_a", "roi_b"])
    top_section("Top targeted EC-lite tests", ec_tests, "q_fdr_targeted_ec", ["target_pathway", "source_roi", "target_roi"])
    top_section("Top pathway-score tests", score_tests, "q_fdr_pathway_scores", ["modality", "target"])
    top_section("Top pathway-behavior correlations", behavior_corr, "q_fdr_behavior_corr", ["modality", "target", "behavior_variable"])

    lines.append("## Family-level summary\n")
    if family_sum.empty:
        lines.append("No family-level summary available.\n")
    else:
        for _, r in family_sum.sort_values("best_q").head(30).iterrows():
            lines.append(
                f"- {r.get('family')}: {r.get('source_table')}; best={r.get('best_description')}; "
                f"quantity={r.get('best_quantity')}; p={r.get('best_p', np.nan):.4g}; q={r.get('best_q', np.nan):.4g}\n"
            )

    lines.append("\n## Scope of downstream modeling\n")
    lines.append(
        "The screening results define a restricted set of candidate pathways for downstream sensitivity analyses. "
        "Selection from the same dataset remains exploratory and post-selection.\n"
    )

    (outdir / "targeted_pathway_fc_ec_behavior_report.md").write_text("".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-activity-summary", required=True)
    parser.add_argument("--fc-edges", required=True)
    parser.add_argument("--ec-edges", required=True)
    parser.add_argument("--behavior-subjects", default="")
    parser.add_argument("--behavior-regex", default=r"(success|reg|up|down|temp|temperature|habituation|slope|sensitivity|pain|rating|order|carry|asym)",
                        help="Regex selecting numeric behavior variables.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    roi_path = Path(args.roi_activity_summary).expanduser().resolve()
    fc_path = Path(args.fc_edges).expanduser().resolve()
    ec_path = Path(args.ec_edges).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    for p in [roi_path, fc_path, ec_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    if outdir.exists() and args.overwrite:
        shutil.rmtree(outdir)
    ensure_dir(outdir)
    ensure_dir(outdir / "tables")

    roi_summary = pd.read_csv(roi_path, sep="\t")
    fc_edges = pd.read_csv(fc_path, sep="\t")
    ec_edges = pd.read_csv(ec_path, sep="\t")

    # Normalize participant identifiers.
    for df in [roi_summary, fc_edges, ec_edges]:
        if "subject" in df.columns:
            df["subject"] = df["subject"].map(clean_subject)

    behavior_path = Path(args.behavior_subjects).expanduser().resolve() if args.behavior_subjects.strip() else None
    behavior = load_behavior(behavior_path, behavior_regex=args.behavior_regex)

    roi_targ = target_roi_activity(roi_summary)
    save_tsv(roi_targ, outdir / "tables" / "targeted_roi_activity_long.tsv")

    roi_tests = test_roi_targets(roi_targ)
    save_tsv(roi_tests, outdir / "tables" / "targeted_roi_activity_tests.tsv")

    fc_targ = target_fc(fc_edges)
    save_tsv(fc_targ, outdir / "tables" / "targeted_fc_edges.tsv")

    fc_tests = test_fc_targets(fc_targ)
    save_tsv(fc_tests, outdir / "tables" / "targeted_fc_tests.tsv")

    ec_targ = target_ec(ec_edges)
    save_tsv(ec_targ, outdir / "tables" / "targeted_ec_edges.tsv")

    ec_tests = test_ec_targets(ec_targ)
    save_tsv(ec_tests, outdir / "tables" / "targeted_ec_tests.tsv")

    scores = build_pathway_scores(roi_targ, fc_targ, ec_targ)
    save_tsv(scores, outdir / "tables" / "pathway_scores.tsv")

    score_tests = test_pathway_scores(scores)
    save_tsv(score_tests, outdir / "tables" / "pathway_score_tests.tsv")

    beh_corr = behavior_correlations(scores, behavior)
    save_tsv(beh_corr, outdir / "tables" / "pathway_behavior_correlations.tsv")

    roi_beh_corr = roi_behavior_correlations(roi_targ, behavior)
    save_tsv(roi_beh_corr, outdir / "tables" / "roi_behavior_correlations.tsv")

    fam_sum = family_summary(score_tests, roi_tests, fc_tests, ec_tests, beh_corr)
    save_tsv(fam_sum, outdir / "tables" / "targeted_family_summary.tsv")

    metadata_base = Path.cwd().resolve()
    manifest = {
        "timestamp_local": now_iso(),
        "script": Path(__file__).name,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "roi_activity_summary": portable_path(roi_path, metadata_base),
        "roi_activity_summary_sha256": sha256_file(roi_path),
        "fc_edges": portable_path(fc_path, metadata_base),
        "fc_edges_sha256": sha256_file(fc_path),
        "ec_edges": portable_path(ec_path, metadata_base),
        "ec_edges_sha256": sha256_file(ec_path),
        "behavior_subjects": portable_path(behavior_path, metadata_base),
        "behavior_subjects_sha256": sha256_file(behavior_path) if behavior_path and behavior_path.exists() else "",
        "outdir": portable_path(outdir, metadata_base),
        "n_targeted_roi_rows": int(len(roi_targ)),
        "n_targeted_fc_rows": int(len(fc_targ)),
        "n_targeted_ec_rows": int(len(ec_targ)),
        "n_pathway_scores": int(len(scores)),
        "n_behavior_subjects": int(behavior["subject"].nunique()) if not behavior.empty else 0,
        "behavior_columns": [c for c in behavior.columns if c != "subject"] if not behavior.empty else [],
        "ec_note": "EC-lite is directed lagged predictive connectivity, not DCM.",
    }
    save_json(manifest, outdir / "manifest.json")

    write_report(outdir, manifest, roi_tests, fc_tests, ec_tests, score_tests, beh_corr, fam_sum)

    print("Targeted pathway screening completed.")
    print(f"Output directory: {outdir}")
    print(f"Targeted ROI rows: {manifest['n_targeted_roi_rows']}")
    print(f"Targeted FC rows: {manifest['n_targeted_fc_rows']}")
    print(f"Targeted EC rows: {manifest['n_targeted_ec_rows']}")
    print(f"Pathway score rows: {manifest['n_pathway_scores']}")
    print(f"Behavior subjects: {manifest['n_behavior_subjects']}")
    print("\nHighest-ranked family summaries:")
    if not fam_sum.empty:
        print(fam_sum.sort_values("best_q").head(20).to_string(index=False))
    else:
        print("No family summary was generated.")


if __name__ == "__main__":
    main()
