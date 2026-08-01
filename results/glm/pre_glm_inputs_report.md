# Pre-GLM first-level input preparation report

## Overview
- Generated: 2026-05-07T19:47:37
- Smoothing inventory: `<PROJECT_ROOT>/derivatives/smoothed_fwhm6/tables/smoothing_inventory.tsv`
- Behavioral latent table: `<PROJECT_ROOT>/derivatives/beh/behavior_computational_analysis/tables/trialwise_behavior_latents.tsv`
- Output directory: `<PROJECT_ROOT>/derivatives/first_level_inputs_analysis`
- FWHM selected: 6.0
- Confound strategy: `standard`
- Include fMRIPrep cosine columns: False
- aCompCor columns requested: 5
- Spike mode: `fd05`; FD spike threshold: 0.5

## Status
- Runs prepared: 288
- READY: 288
- REVIEW_BEFORE_GLM: 0
- FAILED: 0

## Motion-control strategy
- This step does not remove additional runs beyond the already accepted QC/smoothing inventory.
- Motion is handled by retaining QC labels, preparing nuisance regressors, checking confound rank/DOF, and preserving sensitivity-analysis flags.
- Main-analysis policy: include runs marked READY unless subsequent GLM design QC identifies a failure.
- Sensitivity-analysis policy: repeat group-level tests after excluding sensitivity candidates or high-motion subjects/runs.

## Event files
- `events_long.tsv`: thermal-stimulation events plus rating-period events.
- `events_stim.tsv`: one row per thermal-stimulation trial with behavioral latent variables.

## Confounds
- `confounds_standard.tsv` by default includes motion24, white matter, CSF, top aCompCor components, custom FD>0.5 spike regressors, and non-steady-state spikes.
- All fMRIPrep motion_outlier columns are not included by default because they can over-parameterize low-motion runs; use `--spike-mode all` as a sensitivity strategy if needed.
- Cosine regressors are not included by default; use GLM high-pass drift modeling to avoid double high-pass filtering.
- The subsequent GLM stage re-evaluates design rank and exact DOF after adding task regressors and HRF convolution.

## Key outputs
- `tables/first_level_run_inventory.tsv`
- `tables/pre_glm_run_qc.tsv`
- `tables/pre_glm_failed.tsv`
- `tables/confound_strategy_summary.tsv`
- `tables/sensitivity_sets.tsv`
- `sub-XX/run-YY/events_long.tsv`
- `sub-XX/run-YY/events_stim.tsv`
- `sub-XX/run-YY/confounds_standard.tsv`
