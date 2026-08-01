# Methods and reproducibility

## Analysis sequence

1. Behavioral event reconstruction and regulation-success models.
2. Preprocessing QC and retained-run inventory.
3. Run-level and participant-level GLM estimation.
4. Whole-brain, ROI, and multivariate pain-signature analyses.
5. Rating-adjusted trialwise FWL-LSS estimation.
6. Lagged coupling, matched controls, and conditional null models.
7. Six run-separated DCM/PEB sensitivity models.

## Behavioral data

Stimulation events were paired with the subsequent rating event within each run. The complete table contains 3,201 trials from 33 participants: 2,541 passive, 330 up-regulation, and 330 down-regulation trials. Regulation success was defined relative to participant-specific expected pain estimated from passive trials. Positive values indicate instruction-consistent change for both directions.

The state-dependence audit evaluates alternative outcome definitions, cross-fitting, trimming, bounded-scale transformations, and change-score null simulations. Because the simulated null distributions are not centered at zero, the stored summaries report distribution location, central intervals, and the percentile rank of the observed coefficient instead of absolute-value tests against zero.

## Imaging preprocessing and GLM

The reported workflow used fMRIPrep 25.0.0. Of 297 runs, 288 passed the primary movement criterion; all 66 regulation runs were retained. First-level GLMs included pain stimulation, temporal derivatives, temperature modulation, rating periods, motion parameters, aCompCor regressors, spike regressors, and temporal filtering. Run effects were combined within participant by inverse-variance weighting. Group analyses used participant-level maps and voxelwise FDR correction.

## ROI and signature resources

The ROI family contains 32 sensory, insular, control, valuation, subcortical, and descending-modulation masks. Exact labels and voxel counts are in `results/roi_signatures/roi_definitions.tsv`. ROI-family FDR correction was applied across all 32 regions.

The Neurologic Pain Signature weights are not redistributed. Users must obtain the weights independently and supply the image path to the signature scripts. Contrast-map and trialwise FWL-LSS estimates are retained as distinct estimation paths.

## Rating-adjusted FWL-LSS

`code/05_trialwise_connectivity/run_trialwise_roi_connectivity.py` estimates trialwise ROI responses from unsmoothed BOLD data. Each LSS design contains the target trial, other stimulation events, a pooled rating-period nuisance regressor, temporal drift, and retained confounds. The target coefficient is estimated after Frisch–Waugh–Lovell residualization of the target and data with respect to nuisance columns. An SVD pseudoinverse and explicit estimability diagnostics are used.

The reported ROI coupling analysis contains 660 regulation trials from 66 runs and 594 within-run transitions. Transitions never cross run boundaries.

`code/04_roi_signatures/estimate_trialwise_nps.py` applies the same target-trial estimator to voxels with nonzero NPS weights and computes the absolute-weight-normalized expression value.

## Lagged coupling

For every participant and direction, the forward model is:

```text
Target(t+1) ~ Source(t) + Target(t) + Temperature(t+1)
```

Outcome and predictors are standardized within participant. The selected score is the mean of left dlPFC/IFJ coefficients for vmPFC, mOFC, and the combined vmPFC/mOFC target. This association is exploratory and post-selection.

Controls include reverse direction, synchronous zero lag, time reversal, participant bootstrap differences, a shared-source within-run permutation null, and a shared nonzero circular-shift null. The two null models reuse the same within-run transformation across the three overlapping target definitions.

## Run-separated DCM

The DCM workflow uses SPM25 under MATLAB R2023b. Every retained run is modeled independently, so neuronal and hemodynamic states do not cross run boundaries. Each of six five-node models contains 288 run-level DCMs: 222 passive, 33 up-regulation, and 33 down-regulation runs. Across the six models this yields 1,728 DCMs.

Rating periods, a 128-second DCT high-pass basis, and retained confounds enter the nuisance design. Participant-level PEBs estimate PassiveAtMeanRun, UpMinusPassive, DownMinusPassive, RunPosition, DirectionMean, and UpMinusDown effects. Scientific interpretation uses reduced, full-rank group designs and a 90% credible interval or posterior probability threshold of 0.95.

## Environments

Python dependencies are listed in `requirements_analysis.txt` and `requirements_coupling.txt`. Imaging regeneration additionally requires fMRIPrep, Nilearn, nibabel, the external atlas resources, and the NPS weights. DCM regeneration requires MATLAB and SPM25. Dependency release identifiers are retained because they are part of the computational provenance; they are not project-status labels.

## Randomness

Random analyses use explicit numeric seeds. Hash-derived seeds are based on SHA-256 rather than the process-dependent Python `hash()` function. Null distributions and bootstrap iteration counts are stored alongside the reporting source data.
