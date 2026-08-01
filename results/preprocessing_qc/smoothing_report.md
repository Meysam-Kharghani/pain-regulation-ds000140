# Spatial smoothing report

## Overview
- Generated: 2026-05-07T18:54:30
- fMRIPrep directory: `<PROJECT_ROOT>/derivatives/fmriprep`
- QC inclusion table: `<PROJECT_ROOT>/derivatives/qc_fmriprep_analysis/tables/analysis_inclusion_table.tsv`
- Output directory: `<PROJECT_ROOT>/derivatives/smoothed_fwhm6`
- FWHM values: [6.0]
- Include mode: `keep_main`
- Subjects processed: 33
- Runs selected: 288

## Status counts
- OK: 288

## Design policy
- This script only performs spatial smoothing.
- QC decisions are imported from the  fMRIPrep QC table; no new exclusion rule is applied here.
- No PSC conversion is performed.
- No GLM confound file is created here; confound selection and design-rank/DOF checks are performed in the first-level GLM preparation step.
- Default FWHM=6 mm is intended for the main univariate GLM.
- Use unsmoothed fMRIPrep BOLD for pattern-expression, small ROI, NPS, PAG/NAc, or beta-series connectivity analyses.

## Key outputs
- `tables/smoothing_inventory.tsv`
- `tables/smoothing_failed.tsv`
- `tables/qc_rows_used_for_smoothing.tsv`
- Smoothed images in subject/func subfolders.
