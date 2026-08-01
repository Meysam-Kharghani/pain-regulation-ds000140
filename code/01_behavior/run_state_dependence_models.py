#!/usr/bin/env python3
"""
Hierarchical / clustered trial-wise state-dependence model for pain regulation.

Default analysis is robust and dependency-light:
  1) subject fixed-effect trial-wise OLS with cluster-robust SE by subject;
  2) optional MixedLM random-intercept / random-slope model when statsmodels converges;
  3) optional PyMC Bayesian model if --run-pymc is passed and pymc/arviz are installed.

Main outcome: reg_success_z_model on regulation trials.
Positive success means the participant moved ratings in the intended direction
relative to passive expected pain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--derivatives", default=None, help="ds000140 derivatives root")
    p.add_argument("--trialwise", default=None, help="trialwise_model_predictions.tsv")
    p.add_argument("--subject-params", default=None, help="subject_computational_parameters.tsv")
    p.add_argument("--domain-scores", default=None, help="domain_scores.tsv from refined fingerprint")
    p.add_argument("--outdir", default=None, help="Output directory")
    p.add_argument("--run-mixedlm", action="store_true", help="Also try statsmodels MixedLM")
    p.add_argument("--run-pymc", action="store_true", help="Also try optional PyMC Bayesian model")
    p.add_argument("--pymc-draws", type=int, default=1000)
    p.add_argument("--pymc-tune", type=int, default=1000)
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    if args.derivatives is None and (args.trialwise is None or args.subject_params is None):
        raise SystemExit("Provide --derivatives or explicit --trialwise and --subject-params")
    derivatives = Path(args.derivatives) if args.derivatives else None
    trialwise = Path(args.trialwise) if args.trialwise else derivatives / "computational_pain_regulation_analysis" / "tables" / "trialwise_model_predictions.tsv"
    subject_params = Path(args.subject_params) if args.subject_params else derivatives / "computational_pain_regulation_analysis" / "tables" / "subject_computational_parameters.tsv"
    domain_scores = Path(args.domain_scores) if args.domain_scores else derivatives / "refined_individual_fingerprint_analysis" / "tables" / "domain_scores.tsv"
    outdir = Path(args.outdir) if args.outdir else derivatives / "computational_pain_regulation_analysis_hierarchical"
    return {"trialwise": trialwise, "subject_params": subject_params, "domain_scores": domain_scores, "outdir": outdir}


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return x * np.nan
    return (x - x.mean()) / sd


def cluster_ols(formula: str, data: pd.DataFrame, cluster: str):
    import statsmodels.formula.api as smf
    model = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data[cluster]})
    return model


def model_to_rows(name: str, model, terms: List[str]) -> pd.DataFrame:
    rows = []
    for t in terms:
        if t in model.params.index:
            rows.append({
                "model": name,
                "term": t,
                "beta": float(model.params[t]),
                "se": float(model.bse[t]),
                "t_or_z": float(model.tvalues[t]),
                "p": float(model.pvalues[t]),
                "n": int(model.nobs),
                "r2": float(getattr(model, "rsquared", np.nan)),
            })
    return pd.DataFrame(rows)


def fdr_bh(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    pv = p[ok]
    if len(pv) == 0:
        return pd.Series(q, index=pvals.index)
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    vals = ranked * m / (np.arange(m) + 1)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    vals = np.clip(vals, 0, 1)
    out = np.empty_like(vals)
    out[order] = vals
    q[ok] = out
    return pd.Series(q, index=pvals.index)


def fit_mixedlm(data: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    import statsmodels.formula.api as smf
    rows = []
    formula = "reg_success_z_model ~ is_up_c + expected_pain_c + is_up_c:expected_pain_c + global_trial_c + up_first_c"
    for re_formula in ["1", "1 + is_up_c", "1 + is_up_c + expected_pain_c"]:
        try:
            md = smf.mixedlm(formula, data=data, groups=data["subject"], re_formula=re_formula)
            res = md.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
            for term in res.params.index:
                if term in ["Intercept", "is_up_c", "expected_pain_c", "is_up_c:expected_pain_c", "global_trial_c", "up_first_c"]:
                    rows.append({
                        "model": f"MixedLM_re_{re_formula}",
                        "term": term,
                        "beta": float(res.params[term]),
                        "se": float(res.bse[term]) if term in res.bse.index else np.nan,
                        "z": float(res.tvalues[term]) if term in res.tvalues.index else np.nan,
                        "p": float(res.pvalues[term]) if term in res.pvalues.index else np.nan,
                        "aic": float(res.aic),
                        "bic": float(res.bic),
                        "converged": bool(res.converged),
                    })
            # Save random effects from the most complex converged model for downstream participant-level summaries.
            re = pd.DataFrame(res.random_effects).T
            re.index.name = "subject"
            safe_name = re_formula.replace(" ", "").replace("+", "plus")
            re.to_csv(outdir / "tables" / f"mixedlm_random_effects_{safe_name}.tsv", sep="\t")
        except Exception as e:
            rows.append({"model": f"MixedLM_re_{re_formula}", "term": "MODEL_FAILED", "error": str(e)})
    return pd.DataFrame(rows)


def run_optional_pymc(data: pd.DataFrame, outdir: Path, draws: int, tune: int) -> pd.DataFrame:
    try:
        import pymc as pm
        import arviz as az
    except Exception as e:
        return pd.DataFrame([{"model": "PyMC", "term": "PYMC_NOT_AVAILABLE", "error": str(e)}])

    # Keep model small and stable.
    d = data[["subject", "reg_success_z_model", "is_up_c", "expected_pain_c", "global_trial_c", "up_first_c"]].dropna().copy()
    subjects = sorted(d["subject"].unique())
    sid = {s: i for i, s in enumerate(subjects)}
    subj_idx = d["subject"].map(sid).to_numpy(dtype=int)
    y = d["reg_success_z_model"].to_numpy(dtype=float)
    x_up = d["is_up_c"].to_numpy(dtype=float)
    x_exp = d["expected_pain_c"].to_numpy(dtype=float)
    x_int = x_up * x_exp
    x_trial = d["global_trial_c"].to_numpy(dtype=float)
    x_order = d["up_first_c"].to_numpy(dtype=float)
    n_subj = len(subjects)

    with pm.Model() as model:
        sigma_subject = pm.Exponential("sigma_subject", 1.0)
        sigma = pm.Exponential("sigma", 1.0)
        a_raw = pm.Normal("a_raw", 0, 1, shape=n_subj)
        a = pm.Deterministic("subject_intercept", a_raw * sigma_subject)

        beta_up = pm.Normal("beta_up", 0, 1)
        beta_expected = pm.Normal("beta_expected", 0, 1)
        beta_up_x_expected = pm.Normal("beta_up_x_expected", 0, 1)
        beta_trial = pm.Normal("beta_trial", 0, 1)
        beta_order = pm.Normal("beta_order", 0, 1)
        intercept = pm.Normal("intercept", 0, 1)

        mu = intercept + a[subj_idx] + beta_up*x_up + beta_expected*x_exp + beta_up_x_expected*x_int + beta_trial*x_trial + beta_order*x_order
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=4, target_accept=0.9, random_seed=140, progressbar=False)

    az.to_netcdf(idata, outdir / "pymc_hierarchical_state_dependence.nc")
    summ = az.summary(idata, var_names=["intercept", "beta_up", "beta_expected", "beta_up_x_expected", "beta_trial", "beta_order", "sigma_subject", "sigma"], hdi_prob=0.95)
    summ = summ.reset_index().rename(columns={"index": "term"})
    summ.insert(0, "model", "PyMC_random_intercept")
    summ.to_csv(outdir / "tables" / "pymc_posterior_summary.tsv", sep="\t", index=False)
    return summ


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    outdir = paths["outdir"]
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    tw = pd.read_csv(paths["trialwise"], sep="\t")
    subj = pd.read_csv(paths["subject_params"], sep="\t") if paths["subject_params"].exists() else None
    domains = pd.read_csv(paths["domain_scores"], sep="\t") if paths["domain_scores"].exists() else None

    reg = tw[tw["is_reg"].astype(int).eq(1)].copy()
    reg = reg.replace([np.inf, -np.inf], np.nan)
    keep = ["subject", "reg_success_z_model", "is_up", "is_down", "expected_pain_z_passive_model", "global_trial_z_subject", "up_first", "temperature_z_subject"]
    missing = [c for c in keep if c not in reg.columns]
    if missing:
        raise SystemExit(f"Missing required columns in trialwise file: {missing}")
    reg = reg[keep].dropna(subset=["subject", "reg_success_z_model", "is_up", "expected_pain_z_passive_model"]).copy()
    reg["is_up_c"] = reg["is_up"].astype(float) - reg["is_up"].astype(float).mean()
    reg["expected_pain_c"] = zscore(reg["expected_pain_z_passive_model"])
    reg["global_trial_c"] = zscore(reg["global_trial_z_subject"])
    reg["up_first_c"] = reg["up_first"].astype(float) - reg["up_first"].astype(float).mean()
    reg["temperature_c"] = zscore(reg["temperature_z_subject"])

    # Cluster-robust subject fixed-effect model.
    formula_fe = "reg_success_z_model ~ is_up_c + expected_pain_c + is_up_c:expected_pain_c + global_trial_c + up_first_c + C(subject)"
    fe = cluster_ols(formula_fe, reg, "subject")
    terms = ["is_up_c", "expected_pain_c", "is_up_c:expected_pain_c", "global_trial_c", "up_first_c"]
    fe_rows = model_to_rows("cluster_OLS_subject_FE", fe, terms)

    # Model without subject FE but with cluster robust, useful for order effects.
    formula_cluster = "reg_success_z_model ~ is_up_c + expected_pain_c + is_up_c:expected_pain_c + global_trial_c + up_first_c + temperature_c"
    cl = cluster_ols(formula_cluster, reg, "subject")
    cl_rows = model_to_rows("cluster_OLS_no_subject_FE", cl, terms + ["temperature_c"])

    results = pd.concat([fe_rows, cl_rows], ignore_index=True)
    results["q_fdr"] = fdr_bh(results["p"])
    results.to_csv(outdir / "tables" / "hierarchical_state_dependence_cluster_models.tsv", sep="\t", index=False)

    # Derived slopes: down state-dep = expected_pain_c; up state-dep = expected + interaction.
    derived = []
    cov = fe.cov_params()
    b = fe.params
    def lincomb(name: str, weights: Dict[str, float]):
        est = sum(weights.get(k, 0.0)*b.get(k, 0.0) for k in weights)
        var = 0.0
        keys = list(weights.keys())
        for i, ki in enumerate(keys):
            for kj in keys:
                if ki in cov.index and kj in cov.columns:
                    var += weights[ki]*weights[kj]*cov.loc[ki, kj]
        se = float(np.sqrt(var)) if var >= 0 else np.nan
        z = est / se if se and np.isfinite(se) and se > 0 else np.nan
        p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
        derived.append({"contrast": name, "estimate": float(est), "se": se, "z": float(z), "p": float(p), "model": "cluster_OLS_subject_FE"})
    lincomb("down_state_dependence_slope", {"expected_pain_c": 1.0})
    lincomb("up_state_dependence_slope", {"expected_pain_c": 1.0, "is_up_c:expected_pain_c": 1.0})
    lincomb("up_minus_down_state_dependence", {"is_up_c:expected_pain_c": 1.0})
    lincomb("up_minus_down_mean_success", {"is_up_c": 1.0})
    ddf = pd.DataFrame(derived)
    ddf["q_fdr"] = fdr_bh(ddf["p"])
    ddf.to_csv(outdir / "tables" / "hierarchical_up_down_state_contrasts.tsv", sep="\t", index=False)

    # Optional MixedLM.
    if args.run_mixedlm:
        mix = fit_mixedlm(reg, outdir)
        if "p" in mix.columns:
            mix["q_fdr"] = fdr_bh(mix["p"])
        mix.to_csv(outdir / "tables" / "mixedlm_state_dependence_models.tsv", sep="\t", index=False)

    # Optional PyMC.
    if args.run_pymc:
        pymc_summary = run_optional_pymc(reg, outdir, args.pymc_draws, args.pymc_tune)
        pymc_summary.to_csv(outdir / "tables" / "pymc_model_status_or_summary.tsv", sep="\t", index=False)

    # Brain links using  model-derived subject summaries.
    brain_rows = []
    if domains is not None and subj is not None and "subject" in domains.columns:
        merged = subj.merge(domains, on="subject", how="inner")
        comp_cols = [
            "up_success_mean_z", "down_success_mean_z", "mean_reg_success_z", "up_down_asymmetry_z",
            "state_dep_up_slope_expected_pain", "state_dep_down_slope_expected_pain",
            "passive_sensitivity_temp_beta", "passive_precision_inv_rmse", "passive_habituation_global_beta",
        ]
        domain_cols = [c for c in domains.columns if c != "subject" and pd.api.types.is_numeric_dtype(domains[c])]
        for dc in domain_cols:
            for cc in comp_cols:
                if cc not in merged.columns:
                    continue
                x = merged[[dc, cc]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(x) >= 8 and x[dc].std(ddof=1) > 0 and x[cc].std(ddof=1) > 0:
                    r, p = stats.pearsonr(x[dc], x[cc])
                    brain_rows.append({"brain_domain": dc, "computational_parameter": cc, "r": float(r), "p": float(p), "n": int(len(x))})
        bdf = pd.DataFrame(brain_rows)
        if len(bdf):
            bdf["q_fdr"] = fdr_bh(bdf["p"])
            bdf = bdf.sort_values(["p", "brain_domain"])
        bdf.to_csv(outdir / "tables" / "brain_behavior_hierarchical_links.tsv", sep="\t", index=False)

    manifest = {
        "trialwise": str(paths["trialwise"]),
        "subject_params": str(paths["subject_params"]),
        "domain_scores": str(paths["domain_scores"]),
        "outdir": str(outdir),
        "n_reg_trials": int(len(reg)),
        "n_subjects": int(reg["subject"].nunique()),
        "main_model": "reg_success_z_model ~ is_up + passive_expected_pain + is_up:passive_expected_pain + time/order + subject effects",
        "interpretation": {
            "expected_pain_c": "down-regulation state-dependence slope in subject-FE parameterization",
            "expected_pain_c + interaction": "up-regulation state-dependence slope",
            "is_up_c:expected_pain_c": "whether state-dependence differs between up and down",
            "is_up_c": "mean up-vs-down success difference after covariates",
        },
    }
    with open(outdir / "manifest_hierarchical.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Markdown report.
    report = outdir / "hierarchical_state_dependence_analysis_report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Hierarchical state-dependence model - analysis \n\n")
        f.write("## Aim\n")
        f.write("This analysis tests state-dependent pain regulation at the trial level using passive-expected pain and subject-level clustering/hierarchical structure.\n\n")
        f.write("## Data\n")
        f.write(f"- Regulation trials: {len(reg)}\n- Subjects: {reg['subject'].nunique()}\n\n")
        f.write("## Main cluster-robust subject-FE results\n")
        f.write(results.to_markdown(index=False))
        f.write("\n\n## Derived up/down state-dependence contrasts\n")
        f.write(ddf.to_markdown(index=False))
        f.write("\n\n## Interpretation\n")
        f.write("- `expected_pain_c` tests whether success depends on the latent passive pain state for the down/reference condition.\n")
        f.write("- `is_up_c:expected_pain_c` tests whether the state-dependence slope differs between up- and down-regulation.\n")
        f.write("- `is_up_c` tests mean up-vs-down success after controlling for latent passive state, time/order, and subject effects.\n")
        f.write("- Brain-behavior links are exploratory; any inferential use requires explicit multiplicity correction.\n")

    print("Outputs written to", outdir)
    print(results.to_string(index=False))
    print(ddf.to_string(index=False))


if __name__ == "__main__":
    main()
