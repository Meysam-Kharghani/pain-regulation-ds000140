# Trialwise ROI FC/EC report

## Purpose
This analysis estimates trialwise LSS activity for atlas-based ROIs, then tests condition-wise ROI activity, beta-series functional connectivity (FC), and directed lagged beta-series predictive connectivity as an EC-lite analysis.

## Inputs and run status
- Inventory: `<PROJECT_ROOT>/derivatives/first_level_inputs_analysis/tables/first_level_run_inventory.tsv`
- ROI definitions: `<PROJECT_ROOT>/derivatives/roi_masks_analysis_NPS/roi_definitions.tsv`
- BOLD column preference: `bold_unsmoothed`
- Runs attempted: 288
- Runs OK/resumed: 288
- Runs failed: 0
- Trials: 3102
- Subjects: 33
- ROIs in output: 32

## EC interpretation warning
The EC table is a directed lagged predictive-connectivity analysis, not DCM. It is interpreted as an exploratory directed coupling proxy and does not establish causal or physiological directionality.

## Main output files
- `tables/trialwise_roi_lss_long.tsv`
- `tables/roi_activity_subject_tests.tsv`
- `tables/fc_edge_contrast_tests.tsv`
- `tables/ec_lagged_edge_contrast_tests.tsv`
- `tables/pathway_edge_inventory.tsv`

## ROI activity: FDR-significant regulation-related findings
- regulation_mean_minus_passive: reg_SMA_preSMA; mean=197.5, p=0.001332, q=0.04248
- regulation_mean_minus_passive: reg_dlPFC_IFJ_L; mean=237.2, p=0.002655, q=0.04248

## Beta-series FC: FDR-significant regulation-related findings
No FDR-significant regulation-related findings at q < .05.

## Directed lagged EC-lite: FDR-significant regulation-related findings
No FDR-significant regulation-related findings at q < .05.
