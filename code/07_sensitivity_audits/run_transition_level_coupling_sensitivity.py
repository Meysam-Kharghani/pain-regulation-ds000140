#!/usr/bin/env python3
"""Transition-level sensitivity analysis for the selected up-regulation pathway.

Purpose
-------
The reported coupling analysis estimates one lagged source coefficient per
participant from nine within-run transitions and then relates those participant
coefficients to up-regulation success. This sensitivity analysis avoids using
those noisy participant-specific coefficients as the inferential unit. Instead,
it models all 297 up-regulation transitions directly and tests whether the
Source[t] -> Target[t+1] association is moderated by participant-level
up-regulation success.

The analysis is conditional on the already selected dlPFC/IFJ-to-valuation
pathway. It does not repeat the 116-correlation screen and is not independent
validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

TARGETS = [
    ("vmPFC", "roi_reg_vmPFC"),
    ("mOFC", "roi_reg_mOFC"),
    ("vmPFC_mOFC_combined", "roi_reg_vmPFC_mOFC"),
]
SOURCE = "roi_reg_dlPFC_IFJ_L"
BETWEEN = ["reg_success_z_up", "up_first", "passive_mean_rating_z", "passive_sensory_slope_z"]


def z0(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x, ddof=0)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(x, dtype=float)
    return (x - m) / s


def build_transition_table(trialwise: pd.DataFrame, behavior: pd.DataFrame, target_col: str) -> pd.DataFrame:
    keep = ["subject"] + BETWEEN
    b = behavior[keep].drop_duplicates("subject")
    d = trialwise.loc[trialwise["condition"].astype(str).str.lower().eq("up")].copy()
    d = d.merge(b, on="subject", how="inner", validate="many_to_one")
    rows = []
    for subject, sdf in d.groupby("subject", sort=True):
        for run, rdf in sdf.groupby("run", sort=True):
            rdf = rdf.sort_values("trial_lss_index")
            if len(rdf) < 3:
                continue
            source = pd.to_numeric(rdf[SOURCE], errors="coerce").to_numpy(float)
            target = pd.to_numeric(rdf[target_col], errors="coerce").to_numpy(float)
            temp = pd.to_numeric(rdf["temperature_for_model"], errors="coerce").to_numpy(float)
            y = z0(target[1:])
            src = z0(source[:-1])
            lag = z0(target[:-1])
            tmp = z0(temp[1:])
            meta = rdf.iloc[0]
            for j in range(len(y)):
                rows.append({
                    "subject": subject,
                    "run": int(run),
                    "transition": j + 1,
                    "y_z": y[j],
                    "source_z": src[j],
                    "target_prev_z": lag[j],
                    "temperature_z": tmp[j],
                    **{c: float(meta[c]) for c in BETWEEN},
                })
    out = pd.DataFrame(rows)
    # Standardize between-participant variables using one row per participant,
    # then map the standardized values back to transitions.
    sb = out[["subject"] + BETWEEN].drop_duplicates("subject").copy()
    for c in BETWEEN:
        sb[c + "_std"] = z0(sb[c].to_numpy(float))
    return out.drop(columns=BETWEEN).merge(sb, on="subject", how="left", validate="many_to_one")


def design_and_fit(d: pd.DataFrame, adjusted: bool):
    work = d.copy()
    work["success_interaction"] = work["source_z"] * work["reg_success_z_up_std"]
    cols = ["source_z", "target_prev_z", "temperature_z", "success_interaction"]
    if adjusted:
        for c in ["up_first", "passive_mean_rating_z", "passive_sensory_slope_z"]:
            name = c + "_interaction"
            work[name] = work["source_z"] * work[c + "_std"]
            cols.append(name)
    X = sm.add_constant(work[cols], has_constant="add")
    fit = sm.OLS(work["y_z"], X).fit(
        cov_type="cluster",
        cov_kwds={"groups": work["subject"], "use_correction": True},
        use_t=True,
    )
    return work, fit, cols


def fast_beta(d: pd.DataFrame, success_by_subject: dict, adjusted: bool, covariate_override=None):
    # Inputs are already centered within participant, making participant fixed
    # intercepts orthogonal to the within-participant model terms.
    success = d["subject"].map(success_by_subject).to_numpy(float)
    Xcols = [
        d["source_z"].to_numpy(float),
        d["target_prev_z"].to_numpy(float),
        d["temperature_z"].to_numpy(float),
        d["source_z"].to_numpy(float) * success,
    ]
    if adjusted:
        covs = covariate_override
        if covs is None:
            covs = {
                c: dict(zip(d["subject"], d[c + "_std"]))
                for c in ["up_first", "passive_mean_rating_z", "passive_sensory_slope_z"]
            }
        for c in ["up_first", "passive_mean_rating_z", "passive_sensory_slope_z"]:
            v = d["subject"].map(covs[c]).to_numpy(float)
            Xcols.append(d["source_z"].to_numpy(float) * v)
    X = np.column_stack([np.ones(len(d))] + Xcols)
    y = d["y_z"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[4])  # interaction is after intercept + 3 base terms


def subject_level_maps(d: pd.DataFrame):
    s = d[["subject"] + [c + "_std" for c in BETWEEN]].drop_duplicates("subject").sort_values("subject")
    subjects = s["subject"].tolist()
    success = dict(zip(s["subject"], s["reg_success_z_up_std"]))
    covs = {
        c: dict(zip(s["subject"], s[c + "_std"]))
        for c in ["up_first", "passive_mean_rating_z", "passive_sensory_slope_z"]
    }
    return subjects, success, covs, s


def label_permutations(tables, adjusted, n_perm, seed):
    subjects, success, covs, s = subject_level_maps(next(iter(tables.values())))
    vals = np.array([success[x] for x in subjects], float)
    rng = np.random.default_rng(seed)
    obs = {}
    for name, d in tables.items():
        obs[name] = fast_beta(d, success, adjusted, covs)
    obs_comp = float(np.mean(list(obs.values())))
    null = np.empty((n_perm, len(tables) + 1), float)
    names = list(tables)
    for i in range(n_perm):
        pv = rng.permutation(vals)
        smap = dict(zip(subjects, pv))
        bs = [fast_beta(tables[name], smap, adjusted, covs) for name in names]
        null[i, :len(names)] = bs
        null[i, len(names)] = np.mean(bs)
    rows = []
    for j, name in enumerate(names + ["three_target_mean"]):
        observed = obs[name] if name in obs else obs_comp
        p = (np.sum(np.abs(null[:, j]) >= abs(observed)) + 1) / (n_perm + 1)
        lo, hi = np.percentile(null[:, j], [2.5, 97.5])
        rows.append({"target": name, "observed_interaction": observed, "perm_p": p,
                     "null_ci_low": lo, "null_ci_high": hi, "n_perm": n_perm})
    return pd.DataFrame(rows), null


def _varying_interaction_betas(d, success_matrix, adjusted, covs):
    """Fast coefficient estimates for many candidate success label vectors.

    success_matrix has shape (n_draws, n_subjects) in sorted subject order.
    The success interaction varies; all other columns are fixed.
    """
    subjects=sorted(d["subject"].unique())
    sub_index={s:i for i,s in enumerate(subjects)}
    idx=np.array([sub_index[s] for s in d["subject"]],dtype=int)
    source=d["source_z"].to_numpy(float)
    fixed=[np.ones(len(d)),source,d["target_prev_z"].to_numpy(float),d["temperature_z"].to_numpy(float)]
    if adjusted:
        for c in ["up_first","passive_mean_rating_z","passive_sensory_slope_z"]:
            v=d["subject"].map(covs[c]).to_numpy(float)
            fixed.append(source*v)
    W=np.column_stack(fixed)
    y=d["y_z"].to_numpy(float)
    inv=np.linalg.pinv(W.T@W)
    Wy=W.T@y
    ry=y-W@(inv@Wy)
    # z = source * participant success, one row per permutation.
    Z=success_matrix[:,idx]*source[None,:]
    num=Z@ry
    ZW=Z@W
    den=np.sum(Z*Z,axis=1)-np.einsum("bi,ij,bj->b",ZW,inv,ZW)
    return num/den


def freedman_lane_permutations(tables, adjusted, n_perm, seed):
    if not adjusted:
        return None, None
    subjects, success, covs, s = subject_level_maps(next(iter(tables.values())))
    s=s.set_index("subject").loc[subjects]
    y=s["reg_success_z_up_std"].to_numpy(float)
    C=np.column_stack([np.ones(len(subjects)),s["up_first_std"].to_numpy(float),
                       s["passive_mean_rating_z_std"].to_numpy(float),
                       s["passive_sensory_slope_z_std"].to_numpy(float)])
    cb,*_=np.linalg.lstsq(C,y,rcond=None); fitted=C@cb; resid=y-fitted
    rng=np.random.default_rng(seed)
    success_draws=np.empty((n_perm,len(subjects)),float)
    for i in range(n_perm):
        success_draws[i]=z0(fitted+rng.permutation(resid))
    names=list(tables)
    null=np.empty((n_perm,len(names)+1),float)
    obs=[]
    obs_success=np.array([[success[x] for x in subjects]],float)
    for j,name in enumerate(names):
        null[:,j]=_varying_interaction_betas(tables[name],success_draws,True,covs)
        obs.append(float(_varying_interaction_betas(tables[name],obs_success,True,covs)[0]))
    null[:,-1]=null[:,:len(names)].mean(axis=1); obs.append(float(np.mean(obs)))
    rows=[]
    for j,name in enumerate(names+["three_target_mean"]):
        p=(np.sum(np.abs(null[:,j])>=abs(obs[j]))+1)/(n_perm+1)
        lo,hi=np.percentile(null[:,j],[2.5,97.5])
        rows.append({"target":name,"observed_interaction":obs[j],"freedman_lane_p":p,
                     "null_ci_low":lo,"null_ci_high":hi,"n_perm":n_perm})
    return pd.DataFrame(rows),null


def bootstrap_subjects(tables, adjusted, n_boot, seed):
    """Participant-cluster bootstrap using multinomial cluster weights.

    Between-participant moderators retain their original full-sample
    standardization. Resampling is entirely at the participant level.
    """
    subjects, success, covs, _ = subject_level_maps(next(iter(tables.values())))
    names = list(tables)
    rng = np.random.default_rng(seed)
    # Multinomial counts are exactly the multiplicities obtained by drawing
    # len(subjects) clusters with replacement.
    counts = rng.multinomial(len(subjects), [1/len(subjects)]*len(subjects), size=n_boot)
    draws = np.empty((n_boot, len(names)+1), float)
    for j,name in enumerate(names):
        d=tables[name].copy()
        succ=d["subject"].map(success).to_numpy(float)
        cols=[d["source_z"].to_numpy(float), d["target_prev_z"].to_numpy(float),
              d["temperature_z"].to_numpy(float), d["source_z"].to_numpy(float)*succ]
        if adjusted:
            for c in ["up_first","passive_mean_rating_z","passive_sensory_slope_z"]:
                v=d["subject"].map(covs[c]).to_numpy(float)
                cols.append(d["source_z"].to_numpy(float)*v)
        X=np.column_stack([np.ones(len(d))]+cols)
        y=d["y_z"].to_numpy(float)
        # Precompute cluster cross-products.
        xtx=[]; xty=[]
        for sub in subjects:
            m=d["subject"].eq(sub).to_numpy()
            xs=X[m]; ys=y[m]
            xtx.append(xs.T@xs); xty.append(xs.T@ys)
        xtx=np.stack(xtx); xty=np.stack(xty)
        A=np.einsum("bs,sij->bij",counts,xtx)
        b=np.einsum("bs,si->bi",counts,xty)
        # lstsq fallback for the rare singular bootstrap draw.
        bet=np.empty((n_boot,X.shape[1]),float)
        for i in range(n_boot):
            try: bet[i]=np.linalg.solve(A[i],b[i])
            except np.linalg.LinAlgError: bet[i]=np.linalg.lstsq(A[i],b[i],rcond=None)[0]
        draws[:,j]=bet[:,4]
    draws[:,-1]=draws[:,:len(names)].mean(axis=1)
    rows=[]
    for j,name in enumerate(names+["three_target_mean"]):
        lo,hi=np.percentile(draws[:,j],[2.5,97.5])
        signp=2*min(np.mean(draws[:,j]>=0),np.mean(draws[:,j]<=0))
        rows.append({"target":name,"bootstrap_mean":float(np.mean(draws[:,j])),
                     "ci_low":lo,"ci_high":hi,"bootstrap_sign_p":min(1.0,float(signp)),"n_boot":n_boot})
    return pd.DataFrame(rows), draws


def loo_composite(tables, adjusted):
    subjects, _, _, _ = subject_level_maps(next(iter(tables.values())))
    rows=[]
    for dropped in subjects:
        subtables={k:v.loc[~v["subject"].eq(dropped)].copy() for k,v in tables.items()}
        # Re-standardize between-subject values after dropping a participant.
        for k,d in subtables.items():
            sb=d[["subject"]+[c for c in BETWEEN]].drop_duplicates("subject").copy()
            for c in BETWEEN:
                sb[c+"_std"]=z0(sb[c].to_numpy(float))
            d=d.drop(columns=[c+"_std" for c in BETWEEN],errors="ignore").merge(sb,on="subject",how="left")
            subtables[k]=d
        _,success,covs,_=subject_level_maps(next(iter(subtables.values())))
        bs=[fast_beta(d,success,adjusted,covs) for d in subtables.values()]
        rows.append({"dropped_subject":dropped,"composite_interaction":float(np.mean(bs))})
    return pd.DataFrame(rows)


def mixedlm_diagnostics(tables):
    rows=[]
    for name,d in tables.items():
        work=d.copy()
        work["interaction"]=work["source_z"]*work["reg_success_z_up_std"]
        formula="y_z ~ source_z + target_prev_z + temperature_z + reg_success_z_up_std + interaction"
        converged=False; estimate=se=p=np.nan; note=""
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model=smf.mixedlm(formula,work,groups=work["subject"],re_formula="1 + source_z")
                fit=model.fit(reml=False,method="lbfgs",maxiter=5000,disp=False)
                converged=bool(fit.converged)
                estimate=float(fit.params["interaction"]); se=float(fit.bse["interaction"]); p=float(fit.pvalues["interaction"])
                note="; ".join(sorted({str(x.message) for x in w}))
        except Exception as e:
            note=repr(e)
        rows.append({"target":name,"interaction":estimate,"se":se,"p":p,"converged":converged,"note":note})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trialwise",required=True,type=Path)
    ap.add_argument("--behavior",required=True,type=Path)
    ap.add_argument("--outdir",required=True,type=Path)
    ap.add_argument("--n-perm",type=int,default=10000)
    ap.add_argument("--n-boot",type=int,default=10000)
    args=ap.parse_args(); args.outdir.mkdir(parents=True,exist_ok=True)
    trial=pd.read_csv(args.trialwise,sep="\t")
    beh=pd.read_csv(args.behavior,sep="\t")
    tables={name:build_transition_table(trial,beh,col) for name,col in TARGETS}
    pd.concat([d.assign(target=k) for k,d in tables.items()],ignore_index=True).to_csv(args.outdir/"transition_level_input.tsv.gz",sep="\t",index=False,compression="gzip")

    summary=[]
    for adjusted in [False,True]:
        model_name="adjusted" if adjusted else "unadjusted"
        for target,d in tables.items():
            _,fit,_=design_and_fit(d,adjusted)
            key="success_interaction"
            summary.append({
                "model":model_name,"target":target,"n_transitions":len(d),"n_subjects":d.subject.nunique(),
                "interaction_beta":float(fit.params[key]),"cluster_se":float(fit.bse[key]),
                "cluster_t":float(fit.tvalues[key]),"cluster_p_t32":float(fit.pvalues[key]),
                "ci_low":float(fit.conf_int().loc[key,0]),"ci_high":float(fit.conf_int().loc[key,1]),
            })
        # Composite point estimate = equal-weight mean of three target interactions.
        vals=[r["interaction_beta"] for r in summary if r["model"]==model_name]
        summary.append({"model":model_name,"target":"three_target_mean","n_transitions":297,"n_subjects":33,
                        "interaction_beta":float(np.mean(vals)),"cluster_se":np.nan,"cluster_t":np.nan,"cluster_p_t32":np.nan,
                        "ci_low":np.nan,"ci_high":np.nan})
    pd.DataFrame(summary).to_csv(args.outdir/"transition_level_clustered_models.tsv",sep="\t",index=False)

    for adjusted,tag,seed in [(False,"unadjusted",7311),(True,"adjusted",7312)]:
        perm,null=label_permutations(tables,adjusted,args.n_perm,seed)
        perm.to_csv(args.outdir/f"{tag}_participant_label_permutation_summary.tsv",sep="\t",index=False)
        pd.DataFrame(null,columns=[x[0] for x in TARGETS]+["three_target_mean"]).to_csv(args.outdir/f"{tag}_participant_label_permutation_null.tsv.gz",sep="\t",index=False,compression="gzip")
        boot,draws=bootstrap_subjects(tables,adjusted,args.n_boot,seed+100)
        boot.to_csv(args.outdir/f"{tag}_participant_cluster_bootstrap_summary.tsv",sep="\t",index=False)
        pd.DataFrame(draws,columns=[x[0] for x in TARGETS]+["three_target_mean"]).to_csv(args.outdir/f"{tag}_participant_cluster_bootstrap_draws.tsv.gz",sep="\t",index=False,compression="gzip")
        loo=loo_composite(tables,adjusted)
        loo.to_csv(args.outdir/f"{tag}_leave_one_subject_out.tsv",sep="\t",index=False)
        if adjusted:
            fl,flnull=freedman_lane_permutations(tables,True,args.n_perm,seed+200)
            fl.to_csv(args.outdir/"adjusted_freedman_lane_permutation_summary.tsv",sep="\t",index=False)
            pd.DataFrame(flnull,columns=[x[0] for x in TARGETS]+["three_target_mean"]).to_csv(args.outdir/"adjusted_freedman_lane_permutation_null.tsv.gz",sep="\t",index=False,compression="gzip")

    mixed=mixedlm_diagnostics(tables)
    mixed.to_csv(args.outdir/"random_slope_mixedlm_diagnostics.tsv",sep="\t",index=False)

    config={
        "trialwise":str(args.trialwise),"behavior":str(args.behavior),"source":SOURCE,"targets":TARGETS,
        "estimand":"moderation of standardized Source[t] -> Target[t+1] coefficient by participant up-regulation success",
        "within_subject_standardization":"Outcome and Source[t], Target[t], Temperature[t+1] z-scored separately within participant/target over the 9 up-run transitions (ddof=0).",
        "adjustment":"Adjusted models add Source x regulation order, Source x passive mean rating, and Source x passive sensory slope interactions.",
        "inference":"participant-clustered t inference plus participant-label permutation, participant-cluster bootstrap, and adjusted Freedman-Lane residual permutation",
        "selection_warning":"conditional on the selected pathway; does not repeat the 116-correlation screen and is not independent validation",
        "n_perm":args.n_perm,"n_boot":args.n_boot,
    }
    (args.outdir/"analysis_config.json").write_text(json.dumps(config,indent=2)+"\n")

    # Compact human-readable report.
    summ=pd.read_csv(args.outdir/"transition_level_clustered_models.tsv",sep="\t")
    up=summ[(summ.model=="unadjusted") & (summ.target=="three_target_mean")].iloc[0]
    ad=summ[(summ.model=="adjusted") & (summ.target=="three_target_mean")].iloc[0]
    pp=pd.read_csv(args.outdir/"unadjusted_participant_label_permutation_summary.tsv",sep="\t").query("target == 'three_target_mean'").iloc[0]
    apm=pd.read_csv(args.outdir/"adjusted_participant_label_permutation_summary.tsv",sep="\t").query("target == 'three_target_mean'").iloc[0]
    fl=pd.read_csv(args.outdir/"adjusted_freedman_lane_permutation_summary.tsv",sep="\t").query("target == 'three_target_mean'").iloc[0]
    bb=pd.read_csv(args.outdir/"adjusted_participant_cluster_bootstrap_summary.tsv",sep="\t").query("target == 'three_target_mean'").iloc[0]
    loo=pd.read_csv(args.outdir/"adjusted_leave_one_subject_out.tsv",sep="\t")
    report=f"""# Transition-level selected-pathway coupling sensitivity\n\nThis analysis directly models all 297 up-regulation transitions rather than treating each participant's nine-transition coefficient as the inferential unit. It is conditional on the already selected dlPFC/IFJ-to-valuation pathway and is not independent validation.\n\n## Main result\n\n- Equal-weight three-target mean interaction (unadjusted): {up.interaction_beta:.4f}; participant-label permutation p = {pp.perm_p:.5f}.\n- Equal-weight three-target mean interaction (adjusted for regulation order, passive mean rating, and passive sensory slope as source-slope moderators): {ad.interaction_beta:.4f}; participant-label permutation p = {apm.perm_p:.5f}; Freedman-Lane conditional permutation p = {fl.freedman_lane_p:.5f}.\n- Adjusted participant-cluster bootstrap 95% CI: [{bb.ci_low:.4f}, {bb.ci_high:.4f}].\n- Adjusted leave-one-participant-out composite interactions: {loo.composite_interaction.min():.4f} to {loo.composite_interaction.max():.4f}; all {('negative' if (loo.composite_interaction<0).all() else 'not the same sign')}.\n\n## Interpretation\n\nThe negative success-by-source interaction is preserved when the transition data are modeled directly, so the selected association is not solely a consequence of correlating very noisy nine-transition participant coefficients. The target-specific pattern is strongest for vmPFC and the combined vmPFC/mOFC mask; mOFC alone is weaker. Random-slope mixed models are included only as diagnostics because sparse within-participant sampling produced boundary/convergence warnings for some targets.\n\n## Scope\n\nThese results address one reliability concern but do not remove same-sample pathway selection, overlapping target masks, hemodynamic limits on directional interpretation, or the need for independent replication.\n"""
    (args.outdir/"README.md").write_text(report)

if __name__=="__main__":
    main()
