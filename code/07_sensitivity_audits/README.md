# Sensitivity and robustness audits

This directory contains analyses that evaluate robustness and define the
inferential boundaries of the reported results without changing their original
status.

## State-dependence and attenuation

- `run_state_dependence_sensitivity_controls.py` evaluates alternative outcome definitions, cross-fitting, trimming, and stratified analyses.
- `run_state_dependence_coupling_null_controls.py` evaluates bounded-scale and change-score coupling null models.
- `run_down_regulation_attenuation_robustness.py` evaluates ROI and NPS attenuation across motion-sensitivity analysis sets.

## Connectivity and DCM

- `run_matched_coupling_controls.py` reproduces the standardized dlPFC/IFJ-to-vmPFC/mOFC composite and computes matched direction, lag, shuffle, circular-shift, and bootstrap controls.

The selected coupling measure is a three-coefficient composite over overlapping
vmPFC, mOFC, and vmPFC/mOFC masks. The shared-source shuffle is conditional on
the selected pathway and does not repeat the complete screening-and-selection
procedure. The run-separated DCM family is implemented under `code/06_dcm/`
and is interpreted as a model-dependent sensitivity analysis.

## Run-position, NPS, and ROI inference audits

- `run_passive_run_position_controls.py` computes matched passive-run behavioral contrasts and passive-run trends.
- `run_neural_matched_passive_roi.py` computes run-level neural matched-passive ROI contrasts.
- `run_nps_contrast_consistency_audit.py` compares direct-map NPS values with paired and arithmetic condition contrasts.
- `run_nps_participant_influence_audit.py` quantifies participant influence and leave-one-participant-out NPS estimates.
- `run_roi_inference_qc.py` regenerates all-ROI FDR, run and participant counts, and the inferential hierarchy.

Executable commands and verification details are provided in
`docs/reproducibility_workflows.md`.
