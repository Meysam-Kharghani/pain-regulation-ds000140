#!/usr/bin/env python3
"""
Matched standardized coupling controls.

The analysis implements:
1) a matched zero-lag/synchronous control:
   Target[t+1] ~ Source[t+1] + Target[t] + Temperature[t+1]
2) a shared-source within-run shuffle:
   one source permutation per participant/run/iteration, reused for all three targets.
3) a shared non-zero circular source shift:
   one within-run circular shift per participant/run/iteration, reused for all targets,
   preserving the source sequence's cyclic temporal structure.

All participant-level regressions use within-participant z-scoring of outcome
and predictors before OLS, matching the selected repository score.

Outputs also include:
- exact selected-score reproduction check
- participant-bootstrap direct delta-r tests
- Benjamini-Hochberg FDR q-values across the three delta-r tests
- shared-source shuffle null and empirical two-sided P
- software releases and seeds
- compact plotting and tabular summary data
"""

from __future__ import annotations

import argparse
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import statsmodels
import statsmodels.api as sm


def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def read_table(path: Path, nrows=None):
    if str(path).endswith(".gz"):
        return pd.read_csv(path, sep="\t", nrows=nrows)
    if path.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", nrows=nrows)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=nrows)
    raise ValueError(f"Unsupported table: {path}")


def canonical_subject(value):
    s = str(value)
    m = re.search(r"sub[-_]?(\d+)", s, flags=re.I)
    if m:
        return f"sub-{int(m.group(1)):02d}"
    m = re.search(r"(\d+)", s)
    return f"sub-{int(m.group(1)):02d}" if m else s


def infer_col(columns, exact=(), contains=()):
    for col in columns:
        if norm(col) in exact:
            return col
    for col in columns:
        if contains and any(token in norm(col) for token in contains):
            return col
    return None


def zscore_vector(values):
    x = np.asarray(values, dtype=float)
    mean = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x, dtype=float)
    return (x - mean) / sd


def standardized_main_coefficient(y, predictors):
    """
    Return the coefficient for the first predictor after within-regression
    z-scoring of y and every predictor. An intercept is retained.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(predictors, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[mask]
    X = X[mask]
    if len(y) < X.shape[1] + 1:
        return np.nan, len(y)

    yz = zscore_vector(y)
    Xz = np.column_stack([zscore_vector(X[:, j]) for j in range(X.shape[1])])
    design = np.column_stack([np.ones(len(yz)), Xz])
    try:
        beta, *_ = np.linalg.lstsq(design, yz, rcond=None)
        return float(beta[1]), int(len(yz))
    except np.linalg.LinAlgError:
        return np.nan, int(len(yz))


def make_wide_from_long(df, args):
    columns = list(df.columns)
    subject = args.subject_col or infer_col(columns, {"subject", "participant", "sub"}, {"subject"})
    run = args.run_col or infer_col(columns, {"run", "run_number"}, {"run"})
    trial = args.trial_col or infer_col(columns, {"trial", "trial_in_run", "trial_index"}, {"trial"})
    condition = args.condition_col or infer_col(columns, {"condition", "cond", "task_condition"})
    roi = infer_col(columns, {"roi", "region", "name", "mask", "signature"})
    value = infer_col(
        columns,
        {"roi_beta", "value", "estimate", "beta", "mean", "expression", "score"},
        {"roi_beta"},
    )
    temperature = args.temperature_col or infer_col(
        columns,
        {"temperature", "temperature_raw", "stimulus", "stim"},
        {"temp"},
    )

    if all([subject, run, trial, roi, value]):
        id_cols = [subject, run, trial]
        if condition:
            id_cols.append(condition)
        if temperature:
            id_cols.append(temperature)
        tmp = df[id_cols + [roi, value]].copy()
        tmp[value] = pd.to_numeric(tmp[value], errors="coerce")
        wide = (
            tmp.pivot_table(index=id_cols, columns=roi, values=value, aggfunc="mean")
            .reset_index()
        )
        wide.columns = [str(c) for c in wide.columns]
        metadata = {
            "subject": subject,
            "run": run,
            "trial": trial,
            "condition": condition,
            "temperature": temperature,
        }
        return wide, metadata

    return df.copy(), {
        "subject": subject,
        "run": run,
        "trial": trial,
        "condition": condition,
        "temperature": temperature,
    }


def prepare_table(df, args):
    df, metadata = make_wide_from_long(df, args)
    columns = list(df.columns)

    subject = args.subject_col or metadata.get("subject") or infer_col(
        columns, {"subject", "participant", "sub"}, {"subject"}
    )
    run = args.run_col or metadata.get("run") or infer_col(
        columns, {"run", "run_number"}, {"run"}
    )
    trial = args.trial_col or metadata.get("trial") or infer_col(
        columns, {"trial", "trial_in_run", "trial_index"}, {"trial"}
    )
    condition = args.condition_col or metadata.get("condition") or infer_col(
        columns, {"condition", "cond", "task_condition"}
    )
    temperature = args.temperature_col or metadata.get("temperature") or infer_col(
        columns,
        {"temperature", "temperature_raw", "stimulus", "stim"},
        {"temp"},
    )

    if not all([subject, run, trial]):
        raise ValueError(
            f"Could not infer subject/run/trial columns: "
            f"subject={subject}, run={run}, trial={trial}"
        )

    out = df.copy().rename(columns={subject: "subject", run: "run", trial: "trial"})
    out["subject"] = out["subject"].map(canonical_subject)
    out["run"] = pd.to_numeric(out["run"], errors="coerce")
    out["trial"] = pd.to_numeric(out["trial"], errors="coerce")

    if condition and condition in out.columns:
        out = out.rename(columns={condition: "condition"})
    else:
        out["condition"] = "up"

    if temperature and temperature in out.columns and temperature not in {
        "subject", "run", "trial", "condition"
    }:
        out = out.rename(columns={temperature: "temperature"})
    else:
        out["temperature"] = 0.0

    return out


def find_roi_columns(df, args):
    columns = list(df.columns)
    source = args.source_col
    vmpfc = args.vmpfc_col
    mofc = args.mofc_col
    combined = args.combined_col

    if source is None:
        candidates = [
            c for c in columns
            if ("dlpfc" in norm(c) or "ifj" in norm(c))
            and ("_l" in norm(c) or "left" in norm(c) or norm(c).endswith("_l"))
        ]
        source = candidates[0] if candidates else infer_col(
            columns, contains={"dlpfc", "ifj"}
        )

    if vmpfc is None:
        candidates = [
            c for c in columns
            if "vmpfc" in norm(c) and "mofc" not in norm(c)
        ]
        vmpfc = candidates[0] if candidates else infer_col(
            columns, contains={"vmpfc"}
        )

    if mofc is None:
        candidates = [
            c for c in columns
            if "mofc" in norm(c) and "vmpfc" not in norm(c)
        ]
        mofc = candidates[0] if candidates else infer_col(
            columns, contains={"mofc"}
        )

    if combined is None:
        candidates = [
            c for c in columns
            if "vmpfc" in norm(c) and "mofc" in norm(c)
        ]
        combined = candidates[0] if candidates else None

    return source, vmpfc, mofc, combined


def build_transition_arrays(
    subject_df,
    source_col,
    target_col,
    mode,
    shared_permutations=None,
    shared_circular_shifts=None,
):
    y_blocks = []
    x_blocks = []

    for run, run_df in subject_df.groupby("run"):
        run_df = run_df.sort_values("trial")
        source = pd.to_numeric(run_df[source_col], errors="coerce").to_numpy(float)
        target = pd.to_numeric(run_df[target_col], errors="coerce").to_numpy(float)
        temp = pd.to_numeric(
            run_df["temperature"], errors="coerce"
        ).to_numpy(float)

        if len(run_df) < 3:
            continue

        if mode == "forward":
            # Target[t+1] ~ Source[t] + Target[t] + Temperature[t+1]
            y = target[1:]
            X = np.column_stack([source[:-1], target[:-1], temp[1:]])

        elif mode == "reverse":
            # Source[t+1] ~ Target[t] + Source[t] + Temperature[t+1]
            y = source[1:]
            X = np.column_stack([target[:-1], source[:-1], temp[1:]])

        elif mode == "synchronous":
            # Matched zero-lag control:
            # Target[t+1] ~ Source[t+1] + Target[t] + Temperature[t+1]
            y = target[1:]
            X = np.column_stack([source[1:], target[:-1], temp[1:]])

        elif mode == "time_reversed":
            # Target[t] ~ Source[t+1] + Target[t+1] + Temperature[t]
            y = target[:-1]
            X = np.column_stack([source[1:], target[1:], temp[:-1]])

        elif mode == "shared_source_shuffle":
            if shared_permutations is None or run not in shared_permutations:
                raise ValueError("Shared source permutation missing for a run.")
            predictor = source[:-1][shared_permutations[run]]
            y = target[1:]
            X = np.column_stack([predictor, target[:-1], temp[1:]])

        elif mode == "shared_circular_shift":
            if shared_circular_shifts is None or run not in shared_circular_shifts:
                raise ValueError("Shared circular shift missing for a run.")
            shift = int(shared_circular_shifts[run])
            if shift <= 0 or shift >= len(source):
                raise ValueError(
                    f"Circular shift must be in 1..{len(source)-1}; got {shift}."
                )
            shifted_source = np.roll(source, shift)
            y = target[1:]
            X = np.column_stack([
                shifted_source[:-1],
                target[:-1],
                temp[1:],
            ])

        else:
            raise ValueError(f"Unknown mode: {mode}")

        y_blocks.append(y)
        x_blocks.append(X)

    if not y_blocks:
        return np.array([]), np.empty((0, 3))

    return np.concatenate(y_blocks), np.vstack(x_blocks)


def estimate_subject_target(
    subject_df,
    source_col,
    target_col,
    mode,
    shared_permutations=None,
    shared_circular_shifts=None,
):
    y, X = build_transition_arrays(
        subject_df,
        source_col,
        target_col,
        mode,
        shared_permutations=shared_permutations,
        shared_circular_shifts=shared_circular_shifts,
    )
    if len(y) == 0:
        return np.nan, 0
    return standardized_main_coefficient(y, X)


def compute_observed_target_metrics(data, source_col, targets):
    rows = []
    modes = ["forward", "reverse", "synchronous", "time_reversed"]

    for subject, subject_df in data.groupby("subject"):
        for target_name, target_col in targets:
            for mode in modes:
                coef, n_obs = estimate_subject_target(
                    subject_df, source_col, target_col, mode
                )
                rows.append({
                    "subject": subject,
                    "target": target_name,
                    "target_col": target_col,
                    "metric": mode,
                    "coefficient": coef,
                    "n_transitions": n_obs,
                })

    return pd.DataFrame(rows)


def composite_from_target_metrics(target_df):
    composite = (
        target_df
        .pivot_table(
            index=["subject", "metric"],
            values="coefficient",
            aggfunc="mean",
        )
        .reset_index()
        .pivot(index="subject", columns="metric", values="coefficient")
        .reset_index()
    )

    return composite.rename(columns={
        "forward": "forward_composite",
        "reverse": "reverse_composite",
        "synchronous": "synchronous_composite",
        "time_reversed": "time_reversed_composite",
    })


def find_matrix(project):
    candidates = list(project.rglob("screening_predictor_outcome_matrix.tsv"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]


def load_success_matrix(matrix_path):
    matrix = read_table(Path(matrix_path))
    if "subject" not in matrix.columns or "reg_success_z_up" not in matrix.columns:
        raise ValueError(
            "Outcome matrix must contain subject and reg_success_z_up."
        )
    matrix["subject"] = matrix["subject"].map(canonical_subject)
    return matrix


def pearson_metric(data, metric_col):
    subset = data[[metric_col, "reg_success_z_up"]].dropna()
    if (
        len(subset) < 5
        or subset[metric_col].std() == 0
        or subset["reg_success_z_up"].std() == 0
    ):
        return np.nan, np.nan, len(subset)
    r, p = stats.pearsonr(subset[metric_col], subset["reg_success_z_up"])
    return float(r), float(p), int(len(subset))


def observed_correlations(composite, matrix):
    merged = composite.merge(
        matrix[["subject", "reg_success_z_up"]],
        on="subject",
        how="inner",
    )

    rows = []
    for metric in [
        "forward_composite",
        "reverse_composite",
        "synchronous_composite",
        "time_reversed_composite",
    ]:
        r, p, n = pearson_metric(merged, metric)
        rows.append({
            "metric": metric,
            "n": n,
            "pearson_r": r,
            "pearson_p": p,
        })

    return pd.DataFrame(rows), merged


def find_selected_score_column(matrix):
    candidates = [
        c for c in matrix.columns
        if (
            "dlPFC_IFJ_L_to_vmPFC" in c
            or "control_value" in norm(c)
        )
        and "up" in norm(c)
    ]
    return candidates[0] if candidates else None


def selected_score_reproduction(composite, matrix):
    selected_col = find_selected_score_column(matrix)
    if selected_col is None:
        return pd.DataFrame([{
            "check": "selected_score_column_not_found",
            "n": np.nan,
            "value": np.nan,
            "p": np.nan,
        }])

    merged = composite.merge(
        matrix[["subject", selected_col, "reg_success_z_up"]],
        on="subject",
        how="inner",
    )

    rows = []
    for metric in ["forward_composite", selected_col]:
        r, p, n = pearson_metric(merged, metric)
        rows.append({
            "check": f"{metric}_with_up_success",
            "n": n,
            "value": r,
            "p": p,
        })

    subset = merged[["forward_composite", selected_col]].dropna()
    r_rep, p_rep = stats.pearsonr(
        subset["forward_composite"], subset[selected_col]
    )
    rows.append({
        "check": "forward_composite_vs_repository_selected_score",
        "n": len(subset),
        "value": float(r_rep),
        "p": float(p_rep),
    })

    return pd.DataFrame(rows)


def make_shared_permutations(subject_df, rng):
    """
    Create one permutation of Source[t] positions for each run.
    The same permutation is reused across vmPFC, mOFC, and vmPFC/mOFC.
    """
    permutations = {}
    for run, run_df in subject_df.groupby("run"):
        n_transitions = max(len(run_df) - 1, 0)
        permutations[run] = rng.permutation(n_transitions)
    return permutations


def make_shared_circular_shifts(subject_df, rng):
    """
    Draw one non-zero circular shift per run.

    The same shift is reused across vmPFC, mOFC, and vmPFC/mOFC so the
    dependency among the overlapping target definitions is preserved.
    """
    shifts = {}
    for run, run_df in subject_df.groupby("run"):
        n_trials = len(run_df)
        if n_trials < 3:
            continue
        shifts[run] = int(rng.integers(1, n_trials))
    return shifts


def shared_source_circular_shift_null(
    data,
    source_col,
    targets,
    matrix,
    n_shift,
    seed,
):
    rng = np.random.default_rng(seed)
    success = matrix.set_index("subject")["reg_success_z_up"]
    results = []
    grouped_subjects = list(data.groupby("subject"))

    for iteration in range(1, n_shift + 1):
        subject_scores = []

        for subject, subject_df in grouped_subjects:
            shared_shifts = make_shared_circular_shifts(subject_df, rng)
            target_coefficients = []

            for _, target_col in targets:
                coef, _ = estimate_subject_target(
                    subject_df,
                    source_col,
                    target_col,
                    mode="shared_circular_shift",
                    shared_circular_shifts=shared_shifts,
                )
                target_coefficients.append(coef)

            subject_scores.append({
                "subject": subject,
                "circular_shift_composite": np.nanmean(target_coefficients),
            })

        scores = pd.DataFrame(subject_scores)
        scores["reg_success_z_up"] = scores["subject"].map(success)
        valid = scores[["circular_shift_composite", "reg_success_z_up"]].dropna()
        if (
            len(valid) >= 4
            and valid["circular_shift_composite"].std(ddof=1) > 0
            and valid["reg_success_z_up"].std(ddof=1) > 0
        ):
            r = float(stats.pearsonr(
                valid["circular_shift_composite"],
                valid["reg_success_z_up"],
            )[0])
        else:
            r = np.nan

        results.append({
            "iteration": iteration,
            "circular_shift_r_with_success": r,
            "n_subjects": int(len(valid)),
        })

    return pd.DataFrame(results)


def shared_source_shuffle_null(
    data,
    source_col,
    targets,
    matrix,
    n_shuffle,
    seed,
):
    rng = np.random.default_rng(seed)
    success = matrix.set_index("subject")["reg_success_z_up"]
    results = []

    grouped_subjects = list(data.groupby("subject"))

    for iteration in range(1, n_shuffle + 1):
        subject_scores = []

        for subject, subject_df in grouped_subjects:
            shared_permutations = make_shared_permutations(subject_df, rng)
            target_coefficients = []

            for _, target_col in targets:
                coef, _ = estimate_subject_target(
                    subject_df,
                    source_col,
                    target_col,
                    mode="shared_source_shuffle",
                    shared_permutations=shared_permutations,
                )
                target_coefficients.append(coef)

            subject_scores.append({
                "subject": subject,
                "shuffle_composite": np.nanmean(target_coefficients),
            })

        scores = pd.DataFrame(subject_scores)
        scores["reg_success_z_up"] = scores["subject"].map(success)
        r, _, n = pearson_metric(scores, "shuffle_composite")
        results.append({
            "iteration": iteration,
            "shuffle_r_with_success": r,
            "n_subjects": n,
        })

    return pd.DataFrame(results)


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if len(pv) == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[valid] = restored
    return q


def bootstrap_delta_correlations(merged, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    controls = [
        "reverse_composite",
        "synchronous_composite",
        "time_reversed_composite",
    ]
    rows = []

    for control in controls:
        data = merged[
            ["forward_composite", control, "reg_success_z_up"]
        ].dropna().reset_index(drop=True)

        if len(data) < 5:
            continue

        forward_r = stats.pearsonr(
            data["forward_composite"], data["reg_success_z_up"]
        )[0]
        control_r = stats.pearsonr(
            data[control], data["reg_success_z_up"]
        )[0]
        observed_delta = forward_r - control_r

        indices = np.arange(len(data))
        boot_deltas = []

        for _ in range(n_bootstrap):
            sample = data.iloc[
                rng.choice(indices, size=len(indices), replace=True)
            ]
            if (
                sample["forward_composite"].std() == 0
                or sample[control].std() == 0
                or sample["reg_success_z_up"].std() == 0
            ):
                continue

            r_forward = stats.pearsonr(
                sample["forward_composite"],
                sample["reg_success_z_up"],
            )[0]
            r_control = stats.pearsonr(
                sample[control],
                sample["reg_success_z_up"],
            )[0]
            boot_deltas.append(r_forward - r_control)

        boot = np.asarray(boot_deltas, dtype=float)
        count_nonpositive = np.sum(boot <= 0)
        count_nonnegative = np.sum(boot >= 0)
        p_two_sided = min(
            1.0,
            2.0 * min(
                (count_nonpositive + 1) / (len(boot) + 1),
                (count_nonnegative + 1) / (len(boot) + 1),
            ),
        )

        rows.append({
            "comparison": f"forward_minus_{control}",
            "n": len(data),
            "forward_r": forward_r,
            "control_r": control_r,
            "delta_r": observed_delta,
            "bootstrap_ci_low": np.percentile(boot, 2.5),
            "bootstrap_ci_high": np.percentile(boot, 97.5),
            "bootstrap_p_two_sided": p_two_sided,
            "n_bootstrap_valid": len(boot),
        })

    output = pd.DataFrame(rows)
    if not output.empty:
        output["bootstrap_q_fdr_three_comparisons"] = bh_fdr(
            output["bootstrap_p_two_sided"].to_numpy()
        )
    return output


def summarize_null_distribution(values, observed, n_subjects, null_name):
    null_values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if null_values.empty:
        raise ValueError(f"No finite values for {null_name} null.")
    empirical_p = (
        np.sum(np.abs(null_values) >= abs(observed)) + 1
    ) / (len(null_values) + 1)
    return pd.DataFrame([{
        "null_name": null_name,
        "observed_forward_r": observed,
        "n_subjects_observed": n_subjects,
        "n_null": len(null_values),
        "null_mean_r": null_values.mean(),
        "null_sd_r": null_values.std(ddof=1),
        "null_ci_low": np.percentile(null_values, 2.5),
        "null_ci_high": np.percentile(null_values, 97.5),
        "empirical_two_sided_p": empirical_p,
        "observed_percentile_rank": (
            np.sum(null_values <= observed) + 0.5
        ) / (len(null_values) + 1),
    }])


def write_participant_source_data_and_scatter(outdir, merged):
    source = merged[[
        "subject",
        "forward_composite",
        "reverse_composite",
        "synchronous_composite",
        "time_reversed_composite",
        "reg_success_z_up",
    ]].copy()
    source = source.sort_values("subject").reset_index(drop=True)
    source = source.rename(columns={"subject": "participant_id"})
    source.to_csv(
        outdir / "figure4_participant_level_source_data.tsv",
        sep="\t",
        index=False,
    )

    plot_df = source[["forward_composite", "reg_success_z_up"]].dropna()
    if len(plot_df) < 4:
        return

    x = plot_df["forward_composite"].to_numpy(float)
    y = plot_df["reg_success_z_up"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(np.min(x), np.max(x), 200)
    fitted = intercept + slope * grid

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(x, y, s=34, alpha=0.85)
    ax.plot(grid, fitted, linewidth=1.8)
    r, p_value = stats.pearsonr(x, y)
    ax.set_xlabel("Forward coupling composite")
    ax.set_ylabel("Standardized up-regulation success")
    ax.set_title(
        f"Participant-level association (n={len(plot_df)}, "
        f"r={r:.3f}, p={p_value:.4f})"
    )
    ax.axhline(0, linewidth=0.7, alpha=0.35)
    ax.axvline(0, linewidth=0.7, alpha=0.35)
    fig.tight_layout()
    fig.savefig(outdir / "figure4_forward_participant_scatter.png", dpi=300)
    fig.savefig(outdir / "figure4_forward_participant_scatter.svg")
    plt.close(fig)


def write_figure_table_ready(outdir, correlations, shuffle_summary, circular_summary, deltas):
    panel_a = correlations.copy()
    panel_a["panel"] = "A_matched_standardized_metrics"
    panel_a = panel_a.rename(columns={
        "metric": "label",
        "pearson_r": "estimate",
        "pearson_p": "p",
    })
    panel_a["q"] = np.nan
    panel_a["ci_low"] = np.nan
    panel_a["ci_high"] = np.nan

    panel_b = pd.DataFrame([{
        "panel": "B_shared_source_shuffle_null",
        "label": "Observed forward composite",
        "n": shuffle_summary["n_subjects_observed"].iloc[0],
        "estimate": shuffle_summary["observed_forward_r"].iloc[0],
        "p": shuffle_summary["empirical_two_sided_p"].iloc[0],
        "q": np.nan,
        "ci_low": shuffle_summary["null_ci_low"].iloc[0],
        "ci_high": shuffle_summary["null_ci_high"].iloc[0],
    }])

    panel_d = pd.DataFrame([{
        "panel": "D_circular_source_shift_null",
        "label": "Observed forward composite",
        "n": circular_summary["n_subjects_observed"].iloc[0],
        "estimate": circular_summary["observed_forward_r"].iloc[0],
        "p": circular_summary["empirical_two_sided_p"].iloc[0],
        "q": np.nan,
        "ci_low": circular_summary["null_ci_low"].iloc[0],
        "ci_high": circular_summary["null_ci_high"].iloc[0],
    }])

    panel_c = deltas.copy()
    panel_c["panel"] = "C_direct_delta_r_tests"
    panel_c = panel_c.rename(columns={
        "comparison": "label",
        "delta_r": "estimate",
        "bootstrap_p_two_sided": "p",
        "bootstrap_q_fdr_three_comparisons": "q",
        "bootstrap_ci_low": "ci_low",
        "bootstrap_ci_high": "ci_high",
    })

    keep = ["panel", "label", "n", "estimate", "p", "q", "ci_low", "ci_high"]
    combined = pd.concat(
        [panel_a[keep], panel_b[keep], panel_c[keep], panel_d[keep]],
        ignore_index=True,
    )
    combined.to_csv(
        outdir / "coupling_summary_data.tsv",
        sep="\t",
        index=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--trialwise-table", type=Path, required=True)
    parser.add_argument("--matrix-table", type=Path)
    parser.add_argument("--subject-col")
    parser.add_argument("--run-col")
    parser.add_argument("--trial-col")
    parser.add_argument("--condition-col")
    parser.add_argument("--temperature-col")
    parser.add_argument("--source-col")
    parser.add_argument("--vmpfc-col")
    parser.add_argument("--mofc-col")
    parser.add_argument("--combined-col")
    parser.add_argument("--condition-value", default="up")
    parser.add_argument("--n-shuffle", type=int, default=5000)
    parser.add_argument("--n-circular-shift", type=int, default=5000)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--shuffle-seed", type=int, default=1234)
    parser.add_argument("--circular-shift-seed", type=int, default=8241)
    parser.add_argument("--bootstrap-seed", type=int, default=2024)
    parser.add_argument("--minimum-reproduction-r", type=float, default=0.999)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    raw = read_table(args.trialwise_table)
    data = prepare_table(raw, args)

    source, vmpfc, mofc, combined = find_roi_columns(data, args)
    if not source or not vmpfc or not mofc:
        raise ValueError(
            f"Could not infer ROI columns: source={source}, "
            f"vmPFC={vmpfc}, mOFC={mofc}"
        )

    combined_note = "read from table"
    if combined is None:
        combined = "vmPFC_mOFC_combined_generated"
        data[combined] = (
            pd.to_numeric(data[vmpfc], errors="coerce")
            + pd.to_numeric(data[mofc], errors="coerce")
        ) / 2.0
        combined_note = "generated as mean(vmPFC, mOFC)"

    data = data[
        data["condition"].astype(str).str.lower().str.contains(
            str(args.condition_value).lower(),
            na=False,
        )
    ].copy()
    if data.empty:
        raise ValueError(
            f"No rows matched condition '{args.condition_value}'."
        )

    targets = [
        ("vmPFC", vmpfc),
        ("mOFC", mofc),
        ("vmPFC_mOFC_combined", combined),
    ]

    matrix_path = args.matrix_table or find_matrix(args.project)
    if matrix_path is None:
        raise FileNotFoundError(
            "Could not find screening_predictor_outcome_matrix.tsv; "
            "pass --matrix-table explicitly."
        )
    matrix = load_success_matrix(matrix_path)

    target_metrics = compute_observed_target_metrics(
        data, source, targets
    )
    target_metrics.to_csv(
        args.outdir / "matched_standardized_target_coefficients.tsv",
        sep="\t",
        index=False,
    )

    composite = composite_from_target_metrics(target_metrics)
    composite.to_csv(
        args.outdir / "matched_standardized_composite_metrics.tsv",
        sep="\t",
        index=False,
    )

    correlations, merged = observed_correlations(composite, matrix)
    correlations.to_csv(
        args.outdir / "matched_standardized_composite_correlations.tsv",
        sep="\t",
        index=False,
    )
    merged.to_csv(
        args.outdir / "matched_standardized_composite_metrics_with_success.tsv",
        sep="\t",
        index=False,
    )

    reproduction = selected_score_reproduction(composite, matrix)
    reproduction.to_csv(
        args.outdir / "selected_score_reproduction.tsv",
        sep="\t",
        index=False,
    )

    rep_row = reproduction[
        reproduction["check"]
        == "forward_composite_vs_repository_selected_score"
    ]
    if not rep_row.empty:
        reproduction_r = float(rep_row["value"].iloc[0])
        if reproduction_r < args.minimum_reproduction_r:
            raise RuntimeError(
                f"Selected-score reproduction r={reproduction_r:.6f}, "
                f"below required threshold "
                f"{args.minimum_reproduction_r:.6f}."
            )

    shuffle_distribution = shared_source_shuffle_null(
        data=data,
        source_col=source,
        targets=targets,
        matrix=matrix,
        n_shuffle=args.n_shuffle,
        seed=args.shuffle_seed,
    )
    shuffle_distribution.to_csv(
        args.outdir / "shared_source_shuffle_null_distribution.tsv",
        sep="\t",
        index=False,
    )

    forward_r = float(
        correlations.loc[
            correlations["metric"] == "forward_composite",
            "pearson_r",
        ].iloc[0]
    )
    forward_n = int(
        correlations.loc[
            correlations["metric"] == "forward_composite",
            "n",
        ].iloc[0]
    )

    shuffle_summary = summarize_null_distribution(
        shuffle_distribution["shuffle_r_with_success"],
        observed=forward_r,
        n_subjects=forward_n,
        null_name="shared_source_permutation",
    )
    shuffle_summary["null_scope"] = (
        "Conditional on the selected pathway and fixed three-coefficient "
        "composite; the screening/selection stage was not rerun."
    )
    shuffle_summary.to_csv(
        args.outdir / "shared_source_shuffle_null_summary.tsv",
        sep="\t",
        index=False,
    )

    circular_distribution = shared_source_circular_shift_null(
        data=data,
        source_col=source,
        targets=targets,
        matrix=matrix,
        n_shift=args.n_circular_shift,
        seed=args.circular_shift_seed,
    )
    circular_distribution.to_csv(
        args.outdir / "circular_source_shift_null_distribution.tsv",
        sep="\t",
        index=False,
    )
    circular_summary = summarize_null_distribution(
        circular_distribution["circular_shift_r_with_success"],
        observed=forward_r,
        n_subjects=forward_n,
        null_name="within_run_nonzero_circular_source_shift",
    )
    circular_summary["null_scope"] = (
        "Conditional on the selected pathway. One non-zero circular source "
        "shift was drawn per participant and regulation run and reused across "
        "the three overlapping target definitions."
    )
    circular_summary.to_csv(
        args.outdir / "circular_source_shift_null_summary.tsv",
        sep="\t",
        index=False,
    )

    deltas = bootstrap_delta_correlations(
        merged,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed,
    )
    deltas.to_csv(
        args.outdir / "bootstrap_delta_r_with_fdr.tsv",
        sep="\t",
        index=False,
    )

    write_figure_table_ready(
        args.outdir,
        correlations,
        shuffle_summary,
        circular_summary,
        deltas,
    )
    write_participant_source_data_and_scatter(args.outdir, merged)

    pd.DataFrame([{
        "trialwise_table": str(args.trialwise_table),
        "matrix_table": str(matrix_path),
        "source_col": source,
        "vmpfc_col": vmpfc,
        "mofc_col": mofc,
        "combined_col": combined,
        "combined_note": combined_note,
        "standardization": (
            "Within-participant z-scoring of outcome and every predictor "
            "before each subject-target OLS."
        ),
        "forward_model": (
            "Target[t+1] ~ Source[t] + Target[t] + Temperature[t+1]"
        ),
        "reverse_model": (
            "Source[t+1] ~ Target[t] + Source[t] + Temperature[t+1]"
        ),
        "synchronous_model": (
            "Target[t+1] ~ Source[t+1] + Target[t] + Temperature[t+1]"
        ),
        "time_reversed_model": (
            "Target[t] ~ Source[t+1] + Target[t+1] + Temperature[t]"
        ),
        "shuffle_definition": (
            "One within-run Source[t] permutation per subject/run/iteration, "
            "reused across all three overlapping target masks."
        ),
        "circular_shift_definition": (
            "One non-zero circular shift of the full source sequence per "
            "subject/run/iteration, reused across all three targets."
        ),
        "shuffle_seed": args.shuffle_seed,
        "circular_shift_seed": args.circular_shift_seed,
        "bootstrap_seed": args.bootstrap_seed,
        "n_shuffle": args.n_shuffle,
        "n_circular_shift": args.n_circular_shift,
        "n_bootstrap": args.n_bootstrap,
        "target_dependency_note": (
            "vmPFC, mOFC, and vmPFC/mOFC are overlapping/nested masks; "
            "the score is a three-coefficient composite, not three "
            "independent anatomical targets."
        ),
    }]).to_csv(
        args.outdir / "analysis_config_and_seeds.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame([{
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
    }]).to_csv(
        args.outdir / "software_releases.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame([{
        "statement": (
            "Split-half reliability was not estimated because each half "
            "contains only four or five transitions for a four-parameter "
            "regression model."
        )
    }]).to_csv(
        args.outdir / "split_half_not_estimable_note.tsv",
        sep="\t",
        index=False,
    )

    readme = f"""# Matched coupling controls

Selected-score reproduction is reported in `selected_score_reproduction.tsv`.

Use:
- `matched_standardized_composite_correlations.tsv`
- `shared_source_shuffle_null_summary.tsv`
- `circular_source_shift_null_summary.tsv`
- `figure4_participant_level_source_data.tsv`
- `figure4_forward_participant_scatter.png`
- `bootstrap_delta_r_with_fdr.tsv`
- `coupling_summary_data.tsv`

The shared-source shuffle is conditional on the already selected pathway and
does not rerun the screening stage. It is not an independent validation or a
post-selection correction.

Detected columns:
- source: `{source}`
- vmPFC: `{vmpfc}`
- mOFC: `{mofc}`
- combined mask: `{combined}` ({combined_note})
"""
    (args.outdir / "README.md").write_text(
        readme,
        encoding="utf-8",
    )

    print(f"[OK] Matched coupling control outputs written to {args.outdir}")


if __name__ == "__main__":
    main()
