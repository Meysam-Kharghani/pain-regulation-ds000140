# Reproducibility workflows

## Repository validation

```bash
python -m pip install -r requirements_analysis.txt
make validate
```

## Behavioral analyses

```bash
python code/01_behavior/prepare_behavior_table.py --help
python code/01_behavior/run_behavior_models.py --help
python code/01_behavior/run_state_dependence_models.py --help
python code/07_sensitivity_audits/run_state_dependence_coupling_null_controls.py --help
```


## Finite-cluster behavioral inference

Install the R dependencies `clubSandwich`, `fwildclusterboot`, and `dqrng`, then run:

```bash
Rscript code/07_sensitivity_audits/run_behavior_small_sample_inference.R \
  results/behavior/trial_table_order_carryover.tsv.gz \
  results/behavior/trialwise_behavior_latents.tsv \
  derivatives/behavior_small_sample_sensitivity \
  9999 \
  20260803
```

The distributed table is in `results/sensitivity_audits/behavior_small_sample_inference/`.

## Preprocessing QC and GLM

```bash
python code/02_preprocessing_qc/run_preprocessing_qc.py --help
python code/03_glm/prepare_first_level_inputs.py --help
python code/03_glm/run_first_level_glm.py --help
python code/03_glm/run_group_level_glm.py --help
```

## ROI and NPS analyses

```bash
python code/04_roi_signatures/build_roi_masks.py --help
python code/04_roi_signatures/run_roi_signature_analysis.py --help
python code/04_roi_signatures/estimate_trialwise_nps.py --help
python code/04_roi_signatures/run_signature_replication.py --help
```

The restricted NPS weight image must be supplied explicitly.

## Rating-adjusted trialwise ROI estimates

```bash
python code/05_trialwise_connectivity/run_trialwise_roi_connectivity.py \
  --inventory "$DS000140_ROOT/derivatives/first_level_inputs/tables/first_level_run_inventory.tsv" \
  --roi-definitions "$DS000140_ROOT/derivatives/roi_masks/roi_definitions.tsv" \
  --behavior-trials results/behavior/behavior_trials_master_clean.tsv.gz \
  --bold-column bold_unsmoothed \
  --include-rating-nuisance \
  --rating-events-column events_long \
  --outdir "$DS000140_ROOT/derivatives/trialwise_roi_connectivity"
```

## Selected coupling controls

After generating the rating-adjusted trialwise ROI table:

```bash
python code/07_sensitivity_audits/run_matched_coupling_controls.py \
  --project . \
  --trialwise-table "$DS000140_ROOT/derivatives/trialwise_roi_connectivity/tables/trialwise_roi_lss_wide.tsv.gz" \
  --matrix-table results/trialwise_connectivity/screening_predictor_outcome_matrix.tsv \
  --n-shuffle 5000 \
  --n-circular-shift 5000 \
  --n-bootstrap 5000 \
  --outdir "$DS000140_ROOT/derivatives/matched_coupling_controls"
```

The distributed result summary and null distributions are in `results/sensitivity_audits/matched_coupling_controls/` and `results/reporting/`.


## AR(1)-prewhitened ROI and coupling sensitivity

```bash
python code/07_sensitivity_audits/run_ar1_prewhitening_sensitivity.py roi \
  --inventory "$DS000140_ROOT/derivatives/firstlevel_GLM_NATURE_v1/tables/first_level_run_inventory.tsv" \
  --roi-definitions "$DS000140_ROOT/derivatives/roi_masks_NATURE_v3/roi_definitions.tsv" \
  --behavior-trials results/behavior/behavior_trials_master_clean.tsv.gz \
  --bold-column bold_unsmoothed \
  --rating-events-column events_long \
  --runs 3,7 \
  --outdir "$DS000140_ROOT/derivatives/trialwise_roi_connectivity_ar1"

python code/07_sensitivity_audits/run_matched_coupling_controls.py \
  --project . \
  --trialwise-table "$DS000140_ROOT/derivatives/trialwise_roi_connectivity_ar1/tables/trialwise_roi_lss_wide.tsv.gz" \
  --matrix-table results/trialwise_connectivity/screening_predictor_outcome_matrix.tsv \
  --subject-col subject \
  --run-col run \
  --trial-col trial_lss_index \
  --condition-col condition \
  --temperature-col temperature_for_model \
  --source-col roi_reg_dlPFC_IFJ_L \
  --vmpfc-col roi_reg_vmPFC \
  --mofc-col roi_reg_mOFC \
  --combined-col roi_reg_vmPFC_mOFC \
  --condition-value up \
  --n-shuffle 5000 \
  --n-circular-shift 5000 \
  --n-bootstrap 5000 \
  --shuffle-seed 1234 \
  --circular-shift-seed 8241 \
  --bootstrap-seed 2024 \
  --minimum-reproduction-r 0.95 \
  --outdir "$DS000140_ROOT/derivatives/matched_coupling_controls_ar1"
```

Portable trial estimates and derived coupling tables are distributed in
`results/sensitivity_audits/ar1_prewhitening_roi/` and
`results/sensitivity_audits/matched_coupling_controls_ar1/`.

## Run-separated DCM

In MATLAB:

```matlab
setenv('DS000140_ROOT', '/absolute/path/to/ds000140')
setenv('SPM25_PATH', '/absolute/path/to/spm')
addpath('code/06_dcm')
run_all_sessionwise_dcm_variants
```

For one model or a one-participant validation:

```matlab
run_sessionwise_dcm_variant('control_value_descending_primary', 'smoke')
run_sessionwise_dcm_variant('control_value_descending_primary', 'full')
```

## Reporting-source verification

`results/reporting/source_data_map.tsv` maps every figure or table label to its TSV. The repository validator confirms that every mapped file exists and every TSV is rectangular.
