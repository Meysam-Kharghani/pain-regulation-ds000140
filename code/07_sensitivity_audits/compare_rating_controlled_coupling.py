#!/usr/bin/env python3
"""Compare baseline and rating-controlled coupling estimates.

Each input directory must contain outputs from
run_matched_coupling_controls.py, especially:
- matched_standardized_composite_metrics_with_success.tsv
- matched_standardized_target_coefficients.tsv

The comparison is paired at the participant level and reports score stability,
association stability, target-level stability, paired bootstrap intervals, and
publication-ready scatter plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def read_required(directory: Path, filename: str) -> pd.DataFrame:
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def corr(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 4 or frame["x"].std(ddof=1) == 0 or frame["y"].std(ddof=1) == 0:
        return np.nan, np.nan, len(frame)
    result = stats.pearsonr(frame["x"], frame["y"])
    return float(result.statistic), float(result.pvalue), int(len(frame))


def paired_bootstrap_delta_r(
    data: pd.DataFrame,
    baseline_col: str,
    rating_col: str,
    outcome_col: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    frame = data[[baseline_col, rating_col, outcome_col]].dropna().reset_index(drop=True)
    if len(frame) < 5:
        raise ValueError("Too few complete participants for paired bootstrap.")

    r_baseline = stats.pearsonr(frame[baseline_col], frame[outcome_col]).statistic
    r_rating = stats.pearsonr(frame[rating_col], frame[outcome_col]).statistic
    observed = float(r_rating - r_baseline)

    rng = np.random.default_rng(seed)
    boot = []
    indices = np.arange(len(frame))
    for _ in range(n_bootstrap):
        sample = frame.iloc[rng.choice(indices, size=len(indices), replace=True)]
        if any(sample[c].std(ddof=1) == 0 for c in [baseline_col, rating_col, outcome_col]):
            continue
        rb = stats.pearsonr(sample[baseline_col], sample[outcome_col]).statistic
        rr = stats.pearsonr(sample[rating_col], sample[outcome_col]).statistic
        boot.append(float(rr - rb))

    values = np.asarray(boot, dtype=float)
    p_two = min(
        1.0,
        2.0 * min(
            (np.sum(values <= 0) + 1) / (len(values) + 1),
            (np.sum(values >= 0) + 1) / (len(values) + 1),
        ),
    )
    return {
        "n": len(frame),
        "baseline_r_with_success": float(r_baseline),
        "rating_controlled_r_with_success": float(r_rating),
        "delta_r_rating_minus_baseline": observed,
        "bootstrap_ci_low": float(np.percentile(values, 2.5)),
        "bootstrap_ci_high": float(np.percentile(values, 97.5)),
        "bootstrap_p_two_sided": float(p_two),
        "n_bootstrap_valid": int(len(values)),
    }


def scatter(path: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str) -> None:
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(np.min(x), np.max(x), 200)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(x, y, s=34, alpha=0.85)
    ax.plot(grid, intercept + slope * grid, linewidth=1.8)
    r, p_value = stats.pearsonr(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n$n$={len(x)}, $r$={r:.3f}, $p$={p_value:.4f}")
    ax.axhline(0, linewidth=0.7, alpha=0.35)
    ax.axvline(0, linewidth=0.7, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--rating-controlled-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=91027)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    baseline = read_required(
        args.baseline_dir, "matched_standardized_composite_metrics_with_success.tsv"
    )
    rating = read_required(
        args.rating_controlled_dir, "matched_standardized_composite_metrics_with_success.tsv"
    )

    baseline = baseline.rename(columns={
        "forward_composite": "baseline_forward_composite",
        "reverse_composite": "baseline_reverse_composite",
        "synchronous_composite": "baseline_synchronous_composite",
        "time_reversed_composite": "baseline_time_reversed_composite",
    })
    rating = rating.rename(columns={
        "forward_composite": "rating_forward_composite",
        "reverse_composite": "rating_reverse_composite",
        "synchronous_composite": "rating_synchronous_composite",
        "time_reversed_composite": "rating_time_reversed_composite",
    })
    rating = rating.drop(columns=["reg_success_z_up"], errors="ignore")
    merged = baseline.merge(rating, on="subject", how="inner", validate="one_to_one")
    merged.to_csv(args.outdir / "participant_level_baseline_vs_rating_controlled.tsv", sep="\t", index=False)

    rows = []
    for metric in ["forward", "reverse", "synchronous", "time_reversed"]:
        r, p_value, n = corr(
            merged[f"baseline_{metric}_composite"],
            merged[f"rating_{metric}_composite"],
        )
        rows.append({
            "metric": metric,
            "comparison": "baseline_vs_rating_controlled",
            "n": n,
            "r": r,
            "p": p_value,
        })
        rr, rp, rn = corr(
            merged[f"rating_{metric}_composite"], merged["reg_success_z_up"]
        )
        rows.append({
            "metric": metric,
            "comparison": "rating_controlled_vs_up_success",
            "n": rn,
            "r": rr,
            "p": rp,
        })
    pd.DataFrame(rows).to_csv(args.outdir / "correlation_stability_summary.tsv", sep="\t", index=False)

    delta = paired_bootstrap_delta_r(
        merged,
        "baseline_forward_composite",
        "rating_forward_composite",
        "reg_success_z_up",
        args.n_bootstrap,
        args.seed,
    )
    pd.DataFrame([delta]).to_csv(args.outdir / "paired_bootstrap_delta_r.tsv", sep="\t", index=False)

    target_base = read_required(args.baseline_dir, "matched_standardized_target_coefficients.tsv")
    target_rating = read_required(args.rating_controlled_dir, "matched_standardized_target_coefficients.tsv")
    target_base = target_base.rename(columns={"coefficient": "baseline_coefficient"})
    target_rating = target_rating.rename(columns={"coefficient": "rating_controlled_coefficient"})
    keys = ["subject", "target", "metric"]
    target = target_base.merge(
        target_rating[keys + ["rating_controlled_coefficient"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    target.to_csv(args.outdir / "target_level_baseline_vs_rating_controlled.tsv", sep="\t", index=False)

    target_rows = []
    for (target_name, metric), group in target.groupby(["target", "metric"]):
        r, p_value, n = corr(group["baseline_coefficient"], group["rating_controlled_coefficient"])
        target_rows.append({"target": target_name, "metric": metric, "n": n, "r": r, "p": p_value})
    pd.DataFrame(target_rows).to_csv(args.outdir / "target_level_stability_summary.tsv", sep="\t", index=False)

    complete = merged[["baseline_forward_composite", "rating_forward_composite"]].dropna()
    scatter(
        args.outdir / "baseline_vs_rating_controlled_forward",
        complete["baseline_forward_composite"].to_numpy(float),
        complete["rating_forward_composite"].to_numpy(float),
        "Baseline forward composite",
        "Rating-controlled forward composite",
        "Stability of the forward coupling score",
    )
    complete = merged[["rating_forward_composite", "reg_success_z_up"]].dropna()
    scatter(
        args.outdir / "rating_controlled_forward_vs_success",
        complete["rating_forward_composite"].to_numpy(float),
        complete["reg_success_z_up"].to_numpy(float),
        "Rating-controlled forward composite",
        "Standardized up-regulation success",
        "Rating-controlled coupling and up-regulation success",
    )

    print(f"[OK] Comparison outputs written to {args.outdir}")


if __name__ == "__main__":
    main()
