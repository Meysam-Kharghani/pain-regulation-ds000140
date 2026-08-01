# Neural matched-passive ROI analysis

## Inputs

- Trialwise neural table: `results/trialwise_connectivity/trialwise_roi_lss_wide.tsv.gz`
- Matched passive-run table: `matched_passive_run_table.tsv`

## Interpretation

Negative regulation-minus-passive differences indicate lower run-level neural
estimates during regulation than during the selected passive comparator. The
analysis is a run-position sensitivity control and should be interpreted
alongside, rather than in place of, the primary fixed-effect map results.
