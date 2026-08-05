# AR(1)-prewhitened ROI trial estimates

This directory contains the temporal-autocorrelation sensitivity for the 66
regulation runs. The event and nuisance designs match the rating-adjusted
trialwise ROI workflow, while each least-squares-separate model is estimated
with a run-specific AR(1) temporal-noise model.

- `run_inventory_roi_ar1.tsv` records run completion and output dimensions.
- `ar1_rho_by_trial_and_run.tsv.gz` summarizes the AR(1) labels across the 32
  ROI series for each target trial.
- `trialwise_roi_lss_wide.tsv.gz` contains 660 trial estimates for 32 ROIs.
- `analysis_config.json` records portable inputs and software releases.

All 66 runs completed successfully, with 10 trials and 32 ROI estimates per
run. Paths stored in the trial table are dataset-relative.
