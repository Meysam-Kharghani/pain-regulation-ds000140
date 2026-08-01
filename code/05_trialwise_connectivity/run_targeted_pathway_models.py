#!/usr/bin/env python3
"""
run_targeted_pathway_models.py

Reduced post-selection pathway models for ds000140 pain-regulation reanalysis.

Purpose
-------
The targeted pathway screening found that individual differences in directed
lagged EC-lite for the pathway:

  dlPFC/IFJ_L -> vmPFC

were associated with up-regulation and mean regulation success in the targeted screening.
This script runs a smaller post-selection follow-up set with covariate control,
robust SEs, permutation tests, bootstrap CIs, and leave-one-subject-out stability.
These analyses are exploratory and do not constitute independent confirmation.

Important
---------
EC-lite here is directed lagged predictive connectivity, not DCM. It is treated
as an exploratory temporal-prediction metric and does not establish causal or
physiological directionality.

Inputs
------
--pathway-scores:
  targeted_pathway_FC_EC_behavior_analysis/tables/pathway_scores.tsv

--behavior-subjects:
  behavior_computational_analysis/tables/subject_level_behavior_parameters.tsv

Optional:
--model-specs:
  TSV with columns: model_id, predictor_modality, predictor_target,
  predictor_quantity, outcome, family, covariates(optional comma-separated).
  If omitted, the default reduced follow-up model set is used.

Outputs
-------
outdir/
  screening_reduced_pathway_models_report.md
  manifest.json
  tables/
    screening_reduced_pathway_results.tsv
    screening_reduced_pathway_specs.tsv
    screening_reduced_pathway_leave_one_out.tsv
    screening_predictor_outcome_matrix.tsv
    screening_reduced_pathway_family_summary.tsv

Example
-------
python run_targeted_pathway_models.py \
  --pathway-scores <PROJECT_ROOT>/derivatives/targeted_pathway_FC_EC_behavior_analysis/tables/pathway_scores.tsv \
  --behavior-subjects <PROJECT_ROOT>/derivatives/beh/behavior_computational_analysis/tables/subject_level_behavior_parameters.tsv \
  --outdir <PROJECT_ROOT>/derivatives/screening_reduced_pathway_models \
  --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def safe_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def clean_subject(x) -> str:
    s = str(x)
    if s.startswith("sub-"):
        return s
    import re
    m = re.search(r"(\d+)", s)
    return f"sub-{int(m.group(1)):02d}" if m else s


def zscore(s: pd.Series) -> pd.Series:
    x = safe_num(s)
    sd = x.std(skipna=True, ddof=1)
    if pd.isna(sd) or sd < 1e-12:
        return x * np.nan
    return (x - x.mean(skipna=True)) / sd


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


def corr_test(x, y) -> Dict[str, float]:
    d = pd.DataFrame({"x": safe_num(pd.Series(x)), "y": safe_num(pd.Series(y))}).dropna()
    if len(d) < 5 or d["x"].std(ddof=1) < 1e-12 or d["y"].std(ddof=1) < 1e-12:
        return {"n_corr": int(len(d)), "r": np.nan, "p_corr": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    pr = stats.pearsonr(d["x"], d["y"])
    sr = stats.spearmanr(d["x"], d["y"])
    return {
        "n_corr": int(len(d)),
        "r": float(pr.statistic),
        "p_corr": float(pr.pvalue),
        "spearman_rho": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def residualize(y: pd.Series, cov: pd.DataFrame) -> pd.Series:
    d = pd.concat([safe_num(y).rename("y"), cov.apply(safe_num)], axis=1).dropna()
    out = pd.Series(np.nan, index=y.index, dtype=float)
    if len(d) < 5 or cov.shape[1] == 0:
        return safe_num(y)
    X = sm.add_constant(d[cov.columns], has_constant="add")
    fit = sm.OLS(d["y"], X).fit()
    out.loc[d.index] = fit.resid
    return out


def partial_corr_test(x: pd.Series, y: pd.Series, cov: pd.DataFrame) -> Dict[str, float]:
    if cov.shape[1] == 0:
        ct = corr_test(x, y)
        return {"partial_r": ct["r"], "partial_p": ct["p_corr"], "n_partial": ct["n_corr"]}
    d = pd.concat([safe_num(x).rename("x"), safe_num(y).rename("y"), cov.apply(safe_num)], axis=1).dropna()
    if len(d) < max(6, cov.shape[1] + 5):
        return {"partial_r": np.nan, "partial_p": np.nan, "n_partial": int(len(d))}
    rx = residualize(d["x"], d[cov.columns])
    ry = residualize(d["y"], d[cov.columns])
    ct = corr_test(rx, ry)
    return {"partial_r": ct["r"], "partial_p": ct["p_corr"], "n_partial": ct["n_corr"]}


def standardize_design(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in columns:
        out[c] = zscore(df[c])
    return out


def fit_standardized_ols(data: pd.DataFrame, outcome: str, predictor: str, covariates: Sequence[str]) -> Dict[str, object]:
    cols = [outcome, predictor] + list(covariates)
    d = data[cols].copy()
    for c in cols:
        d[c] = safe_num(d[c])
    d = d.dropna()

    out: Dict[str, object] = {"n": int(len(d)), "covariates_used": ",".join(covariates)}
    if len(d) < max(8, len(covariates) + 5):
        out.update({"std_beta": np.nan, "se_hc3": np.nan, "t_hc3": np.nan, "p_hc3": np.nan, "r2": np.nan})
        return out

    zd = standardize_design(d, cols)
    X = sm.add_constant(zd[[predictor] + list(covariates)], has_constant="add")
    y = zd[outcome]
    fit = sm.OLS(y, X).fit(cov_type="HC3")

    out.update({
        "std_beta": float(fit.params[predictor]),
        "se_hc3": float(fit.bse[predictor]),
        "t_hc3": float(fit.tvalues[predictor]),
        "p_hc3": float(fit.pvalues[predictor]),
        "r2": float(fit.rsquared),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
    })
    return out


def bootstrap_beta(data: pd.DataFrame, outcome: str, predictor: str, covariates: Sequence[str], n_boot: int, seed: int) -> Dict[str, object]:
    cols = [outcome, predictor] + list(covariates)
    d = data[cols].copy()
    for c in cols:
        d[c] = safe_num(d[c])
    d = d.dropna()
    if len(d) < max(8, len(covariates) + 5):
        return {"boot_n": 0, "boot_beta_mean": np.nan, "boot_ci_lo": np.nan, "boot_ci_hi": np.nan, "boot_p_sign": np.nan}

    rng = np.random.default_rng(seed)
    betas = []
    for _ in range(n_boot):
        sample_idx = rng.choice(d.index.to_numpy(), size=len(d), replace=True)
        dd = d.loc[sample_idx].reset_index(drop=True)
        try:
            res = fit_standardized_ols(dd, outcome, predictor, covariates)
            b = res.get("std_beta", np.nan)
            if np.isfinite(b):
                betas.append(float(b))
        except Exception:
            continue
    betas = np.array(betas, dtype=float)
    if len(betas) < 20:
        return {"boot_n": int(len(betas)), "boot_beta_mean": np.nan, "boot_ci_lo": np.nan, "boot_ci_hi": np.nan, "boot_p_sign": np.nan}
    ci = np.percentile(betas, [2.5, 97.5])
    # Two-sided sign/bootstrap probability around zero.
    p_sign = 2 * min(np.mean(betas >= 0), np.mean(betas <= 0))
    return {
        "boot_n": int(len(betas)),
        "boot_beta_mean": float(np.mean(betas)),
        "boot_ci_lo": float(ci[0]),
        "boot_ci_hi": float(ci[1]),
        "boot_p_sign": float(min(1.0, p_sign)),
    }


def permutation_p(data: pd.DataFrame, outcome: str, predictor: str, covariates: Sequence[str], observed_beta: float, n_perm: int, seed: int) -> Dict[str, object]:
    cols = [outcome, predictor] + list(covariates)
    d = data[cols].copy()
    for c in cols:
        d[c] = safe_num(d[c])
    d = d.dropna()
    if len(d) < max(8, len(covariates) + 5) or not np.isfinite(observed_beta):
        return {"perm_n": 0, "perm_p": np.nan}
    rng = np.random.default_rng(seed)
    betas = []
    for _ in range(n_perm):
        dd = d.copy()
        dd[predictor] = rng.permutation(dd[predictor].to_numpy())
        try:
            b = fit_standardized_ols(dd, outcome, predictor, covariates).get("std_beta", np.nan)
            if np.isfinite(b):
                betas.append(float(b))
        except Exception:
            continue
    betas = np.array(betas, dtype=float)
    if len(betas) == 0:
        return {"perm_n": 0, "perm_p": np.nan}
    p = (np.sum(np.abs(betas) >= abs(observed_beta)) + 1) / (len(betas) + 1)
    return {"perm_n": int(len(betas)), "perm_p": float(p)}


def leave_one_out(data: pd.DataFrame, outcome: str, predictor: str, covariates: Sequence[str]) -> Dict[str, object]:
    cols = ["subject", outcome, predictor] + list(covariates)
    d = data[cols].copy()
    for c in cols:
        if c != "subject":
            d[c] = safe_num(d[c])
    d = d.dropna()
    betas = []
    dropped = []
    for sub in d["subject"].unique():
        dd = d[d["subject"] != sub].copy()
        try:
            b = fit_standardized_ols(dd, outcome, predictor, covariates).get("std_beta", np.nan)
        except Exception:
            b = np.nan
        if np.isfinite(b):
            betas.append(float(b)); dropped.append(sub)
    betas = np.array(betas, dtype=float)
    if len(betas) == 0:
        return {"loo_n": 0, "loo_beta_min": np.nan, "loo_beta_max": np.nan, "loo_beta_mean": np.nan, "loo_same_sign_pct": np.nan}
    full_beta = fit_standardized_ols(d, outcome, predictor, covariates).get("std_beta", np.nan)
    if np.isfinite(full_beta) and abs(full_beta) > 0:
        same = np.mean(np.sign(betas) == np.sign(full_beta))
    else:
        same = np.nan
    return {
        "loo_n": int(len(betas)),
        "loo_beta_min": float(np.min(betas)),
        "loo_beta_max": float(np.max(betas)),
        "loo_beta_mean": float(np.mean(betas)),
        "loo_same_sign_pct": float(100 * same) if np.isfinite(same) else np.nan,
    }


def default_model_specs() -> pd.DataFrame:
    rows = []
    # Primary EC-lite behavioral confirmation.
    for q in ["ec_up", "up_minus_passive"]:
        for out in ["reg_success_z_up", "reg_success_raw_up", "mean_reg_success_z", "mean_reg_success_raw"]:
            rows.append({
                "model_id": f"primary_ec_dlpfc_vmPFC_{q}_to_{out}",
                "family": "primary_control_value_EC_behavior",
                "predictor_modality": "EC_lite_directed_lagged",
                "predictor_target": "dlPFC_IFJ_L_to_vmPFC_control_value",
                "predictor_quantity": q,
                "outcome": out,
                "covariates": "up_first,passive_mean_rating_z,passive_sensory_slope_z",
                "priority": "primary",
            })
    # Secondary ROI control engagement.
    for target in ["control_dlPFC_IFJ_L", "control_SMA_preSMA"]:
        for q in ["regulation_mean_minus_passive", "up_minus_passive"]:
            for out in ["reg_success_z_up", "mean_reg_success_z"]:
                rows.append({
                    "model_id": f"secondary_roi_{target}_{q}_to_{out}",
                    "family": "secondary_control_activity_behavior",
                    "predictor_modality": "ROI_activity",
                    "predictor_target": target,
                    "predictor_quantity": q,
                    "outcome": out,
                    "covariates": "up_first,passive_mean_rating_z,passive_sensory_slope_z",
                    "priority": "secondary",
                })
    # Secondary FC sensory/interoceptive pathway, mainly for down-regulation.
    for out in ["reg_success_z_down", "mean_reg_success_z"]:
        rows.append({
            "model_id": f"secondary_fc_sensory_thalamus_insula_down_to_{out}",
            "family": "secondary_sensory_FC_behavior",
            "predictor_modality": "FC_beta_series",
            "predictor_target": "sensory_thalamus_S1_S2_insula",
            "predictor_quantity": "down_minus_passive",
            "outcome": out,
            "covariates": "up_first,passive_mean_rating_z,passive_sensory_slope_z",
            "priority": "secondary",
        })
    return pd.DataFrame(rows)


def load_pathway_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = ["subject", "modality", "target", "quantity", "score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"pathway_scores missing columns: {missing}")
    df = df.copy()
    df["subject"] = df["subject"].map(clean_subject)
    df["score"] = safe_num(df["score"])
    return df


def load_behavior(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "subject" not in df.columns:
        raise ValueError("behavior file must contain subject column")
    df = df.copy()
    df["subject"] = df["subject"].map(clean_subject)
    for c in df.columns:
        if c != "subject":
            df[c] = safe_num(df[c])
    return df.drop_duplicates("subject")


def make_predictor_matrix(scores: pd.DataFrame, specs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    subjects = sorted(scores["subject"].unique())
    mat = pd.DataFrame({"subject": subjects})
    name_map = {}
    for _, s in specs.iterrows():
        col = f"pred_{s['predictor_modality']}_{s['predictor_target']}_{s['predictor_quantity']}"
        col = col.replace(" ", "_")
        if col in mat.columns:
            name_map[s["model_id"]] = col
            continue
        d = scores[
            (scores["modality"].astype(str) == str(s["predictor_modality"])) &
            (scores["target"].astype(str) == str(s["predictor_target"])) &
            (scores["quantity"].astype(str) == str(s["predictor_quantity"]))
        ][["subject", "score"]].copy()
        d = d.rename(columns={"score": col})
        mat = mat.merge(d, on="subject", how="left")
        name_map[s["model_id"]] = col
    return mat, name_map


def parse_covariates(spec_cov: str, default_covariates: Sequence[str], data_cols: Sequence[str]) -> List[str]:
    if pd.notna(spec_cov) and str(spec_cov).strip():
        covs = [c.strip() for c in str(spec_cov).split(",") if c.strip()]
    else:
        covs = list(default_covariates)
    return [c for c in covs if c in data_cols]


def run_models(scores: pd.DataFrame, behavior: pd.DataFrame, specs: pd.DataFrame,
               default_covariates: Sequence[str], n_boot: int, n_perm: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_mat, name_map = make_predictor_matrix(scores, specs)
    data = behavior.merge(pred_mat, on="subject", how="inner")

    rows = []
    loo_rows = []
    rng = np.random.default_rng(seed)

    for _, s in specs.iterrows():
        model_id = s["model_id"]
        predictor = name_map.get(model_id)
        outcome = str(s["outcome"])
        covariates = parse_covariates(str(s.get("covariates", "")), default_covariates, data.columns)

        base = s.to_dict()
        base.update({"predictor_column": predictor, "covariates_used": ",".join(covariates)})

        if predictor not in data.columns or outcome not in data.columns:
            row = dict(base)
            row.update({"status": "MISSING_PREDICTOR_OR_OUTCOME", "n": 0})
            rows.append(row)
            continue

        dcols = ["subject", outcome, predictor] + covariates
        d = data[dcols].copy().dropna()
        row = dict(base)
        row.update({"status": "OK" if len(d) >= max(8, len(covariates)+5) else "LOW_N", "n_complete": int(len(d))})

        # Zero-order and partial correlation.
        row.update(corr_test(d[predictor], d[outcome]))
        pc = partial_corr_test(d[predictor], d[outcome], d[covariates] if covariates else pd.DataFrame(index=d.index))
        row.update(pc)

        # Standardized OLS with robust SE.
        fit = fit_standardized_ols(d, outcome, predictor, covariates)
        row.update(fit)

        if row.get("status") == "OK" and np.isfinite(row.get("std_beta", np.nan)):
            local_seed = int(rng.integers(0, 2**31-1))
            row.update(bootstrap_beta(d, outcome, predictor, covariates, n_boot=n_boot, seed=local_seed))
            row.update(permutation_p(d, outcome, predictor, covariates, observed_beta=row["std_beta"], n_perm=n_perm, seed=local_seed + 1))
            loo = leave_one_out(d, outcome, predictor, covariates)
            row.update(loo)
            loo_row = dict(base); loo_row.update(loo); loo_rows.append(loo_row)
        rows.append(row)

    results = pd.DataFrame(rows)
    if not results.empty:
        results = fdr(results, "p_hc3", "q_fdr_hc3_by_priority", by=["priority"])
        results = fdr(results, "partial_p", "q_fdr_partial_by_priority", by=["priority"])
        results = fdr(results, "perm_p", "q_fdr_perm_by_priority", by=["priority"])
    return results, pd.DataFrame(loo_rows), data


def family_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    for (priority, family), d in results.groupby(["priority", "family"], dropna=False):
        dd = d[pd.to_numeric(d.get("q_fdr_hc3_by_priority", np.nan), errors="coerce").notna()].copy()
        if dd.empty:
            continue
        best = dd.sort_values("q_fdr_hc3_by_priority").iloc[0].to_dict()
        rows.append({
            "priority": priority,
            "family": family,
            "best_model_id": best.get("model_id", ""),
            "best_outcome": best.get("outcome", ""),
            "best_predictor_target": best.get("predictor_target", ""),
            "best_predictor_quantity": best.get("predictor_quantity", ""),
            "best_std_beta": best.get("std_beta", np.nan),
            "best_p_hc3": best.get("p_hc3", np.nan),
            "best_q_hc3": best.get("q_fdr_hc3_by_priority", np.nan),
            "best_partial_r": best.get("partial_r", np.nan),
            "best_partial_p": best.get("partial_p", np.nan),
            "best_perm_p": best.get("perm_p", np.nan),
            "boot_ci_lo": best.get("boot_ci_lo", np.nan),
            "boot_ci_hi": best.get("boot_ci_hi", np.nan),
            "loo_same_sign_pct": best.get("loo_same_sign_pct", np.nan),
            "n_complete": best.get("n_complete", np.nan),
        })
    return pd.DataFrame(rows)


def write_report(outdir: Path, manifest: Dict, results: pd.DataFrame, fam: pd.DataFrame) -> None:
    lines = []
    lines.append("# Reduced pathway follow-up models report\n\n")
    lines.append("## Purpose\n")
    lines.append("This analysis applies a reduced post-selection set of pathway-behavior models after the targeted screening. It evaluates whether individual differences in the selected lagged control-value composite are associated with regulation success after covariate adjustment; it is not an independent confirmation.\n\n")
    lines.append("## Inputs\n")
    lines.append(f"- Pathway scores: `{manifest['pathway_scores']}`\n")
    lines.append(f"- Behavior subjects: `{manifest['behavior_subjects']}`\n")
    lines.append(f"- Output directory: `{manifest['outdir']}`\n")
    lines.append(f"- Default covariates: `{', '.join(manifest['default_covariates'])}`\n")
    lines.append(f"- Bootstrap samples: {manifest['n_boot']}\n")
    lines.append(f"- Permutations: {manifest['n_perm']}\n\n")
    lines.append("## EC interpretation warning\n")
    lines.append("EC-lite is directed lagged predictive connectivity, not DCM. It is treated as an exploratory temporal-prediction metric and does not establish causal or physiological directionality.\n\n")

    if not fam.empty:
        lines.append("## Family-level summary\n")
        for _, r in fam.sort_values("best_q_hc3").iterrows():
            lines.append(
                f"- {r['priority']} | {r['family']}: {r['best_model_id']} | beta={r['best_std_beta']:.4g}, "
                f"p={r['best_p_hc3']:.4g}, q={r['best_q_hc3']:.4g}, partial_r={r['best_partial_r']:.4g}, "
                f"perm_p={r['best_perm_p']:.4g}, bootstrap CI=[{r['boot_ci_lo']:.4g}, {r['boot_ci_hi']:.4g}], "
                f"LOO same sign={r['loo_same_sign_pct']:.1f}%\n"
            )
        lines.append("\n")

    if not results.empty:
        lines.append("## Primary models\n")
        d = results[results["priority"].astype(str) == "primary"].copy()
        if d.empty:
            lines.append("No primary model rows found.\n\n")
        else:
            d = d.sort_values("q_fdr_hc3_by_priority")
            for _, r in d.iterrows():
                lines.append(
                    f"- {r.get('model_id')}: outcome={r.get('outcome')}, predictor={r.get('predictor_quantity')} "
                    f"({r.get('predictor_target')}); beta={r.get('std_beta', np.nan):.4g}, "
                    f"p={r.get('p_hc3', np.nan):.4g}, q={r.get('q_fdr_hc3_by_priority', np.nan):.4g}, "
                    f"partial_r={r.get('partial_r', np.nan):.4g}, perm_p={r.get('perm_p', np.nan):.4g}\n"
                )
            lines.append("\n")

    lines.append("## Interpretation guide\n")
    lines.append("Concordance among the HC3 p-value, partial correlation, permutation p-value, bootstrap CI, and leave-one-participant-out sign stability strengthens the descriptive evidence. Attenuation after covariate adjustment reinforces the exploratory interpretation.\n")
    (outdir / "screening_reduced_pathway_models_report.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathway-scores", required=True)
    ap.add_argument("--behavior-subjects", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model-specs", default="")
    ap.add_argument("--default-covariates", default="up_first,passive_mean_rating_z,passive_sensory_slope_z")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260607)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    score_path = Path(args.pathway_scores).expanduser().resolve()
    beh_path = Path(args.behavior_subjects).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    if not beh_path.exists():
        raise FileNotFoundError(beh_path)
    if outdir.exists() and args.overwrite:
        shutil.rmtree(outdir)
    ensure_dir(outdir); ensure_dir(outdir / "tables")

    scores = load_pathway_scores(score_path)
    behavior = load_behavior(beh_path)
    if args.model_specs.strip():
        specs = pd.read_csv(Path(args.model_specs).expanduser().resolve(), sep="\t")
    else:
        specs = default_model_specs()
    default_cov = [c.strip() for c in args.default_covariates.split(",") if c.strip()]

    results, loo, matrix = run_models(scores, behavior, specs, default_cov, args.n_boot, args.n_perm, args.seed)
    fam = family_summary(results)

    save_tsv(specs, outdir / "tables" / "screening_reduced_pathway_specs.tsv")
    save_tsv(matrix, outdir / "tables" / "screening_predictor_outcome_matrix.tsv")
    save_tsv(results, outdir / "tables" / "screening_reduced_pathway_results.tsv")
    save_tsv(loo, outdir / "tables" / "screening_reduced_pathway_leave_one_out.tsv")
    save_tsv(fam, outdir / "tables" / "screening_reduced_pathway_family_summary.tsv")

    manifest = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pathway_scores": str(score_path),
        "pathway_scores_sha256": sha256_file(score_path),
        "behavior_subjects": str(beh_path),
        "behavior_subjects_sha256": sha256_file(beh_path),
        "model_specs": str(Path(args.model_specs).expanduser().resolve()) if args.model_specs.strip() else "default_internal_specs",
        "outdir": str(outdir),
        "default_covariates": default_cov,
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "n_models": int(len(specs)),
        "n_subjects_behavior": int(behavior["subject"].nunique()),
        "ec_note": "EC-lite is directed lagged predictive connectivity, not DCM.",
    }
    save_json(manifest, outdir / "manifest.json")
    write_report(outdir, manifest, results, fam)

    print("Processing completed successfully.")
    print(f"Output directory: {outdir}")
    print(f"Models: {len(specs)}")
    if not fam.empty:
        print("\nFamily summary:")
        print(fam.sort_values("best_q_hc3").to_string(index=False))
    if not results.empty:
        print("\nTop primary models:")
        cols = ["model_id", "outcome", "predictor_quantity", "std_beta", "p_hc3", "q_fdr_hc3_by_priority", "partial_r", "partial_p", "perm_p", "boot_ci_lo", "boot_ci_hi", "loo_same_sign_pct", "n_complete"]
        print(results[results["priority"].astype(str)=="primary"].sort_values("q_fdr_hc3_by_priority")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
