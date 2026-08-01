# NPS original-style replication report

## Purpose
This analysis tests whether the updated NPS results reproduce the original paper's core pattern: strong NPS sensitivity to stimulus/temperature, but no canonical self-regulation effect in the up-versus-down contrast. It also explicitly tests the new down-minus-passive finding.

## Inputs
- Signature values: `<PROJECT_ROOT>/derivatives/roi_signature_analysis_with_nps/tables/signature_values.tsv`
- Signature: `NPS`
- Measures: `signature_weighted_mean, signature_dot`
- Sets: `fd_bad_run_exclude, high_motion_subject_exclude, main`

## Important limitation
The original study used single-trial NPS dot-products in multilevel models. This script uses subject-level first-level GLM effect/contrast maps, so it is a replication-style analysis rather than an exact reproduction of the original trial-level model.

## Primary summary: main, signature_weighted_mean
- Passive temperature NPS effect: mean=0.00701, p=4.74e-11
- Canonical up-minus-down NPS effect: mean=0.02709, p=0.3578
- Up-minus-passive NPS effect: mean=0.01201, p=0.5127
- Down-minus-passive NPS effect: mean=-0.00484, p=9.72e-05
- Regmean-minus-passive NPS effect: mean=0.01319, p=0.4794
- Original-like canonical pattern: `True`
- Down-specific divergence: `True`
- Interpretation: Original-like on canonical up-vs-down test; also shows down-specific NPS reduction

## Interpretation
The subject-level contrast-map path is consistent with a null canonical up-minus-down NPS effect while showing a negative down-minus-passive estimate. The rating-adjusted FWL-LSS analysis reported under `results/reporting` does not reproduce the down-minus-passive effect. Together, these results indicate analysis-path dependence rather than a general NPS effect of regulation.

## Output tables
- `tables/nps_direct_effect_tests.tsv`
- `tables/nps_condition_paired_tests.tsv`
- `tables/nps_temperature_slope_tests.tsv`
- `tables/nps_original_style_summary.tsv`
- `tables/nps_subject_condition_values.tsv`
