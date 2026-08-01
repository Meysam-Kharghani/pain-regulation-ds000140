# fMRIPrep confounds QC report

## Input
- Generated: 2026-05-07T10:10:35
- fMRIPrep directory: `<PROJECT_ROOT>/derivatives/fmriprep`
- Number of confounds files: 297
- Number of subjects: 33
- Regulation runs: 3,7

## Decision counts
- EXCLUDE: 9
- KEEP: 56
- KEEP_FOR_MAIN_BUT_SENSITIVITY_EXCLUDE_CANDIDATE: 125
- KEEP_WITH_WARNING: 107

## Motion summary
- Mean FD across runs: mean=0.1646, median=0.1381, max=0.5391
- Percent FD>0.5 across runs: mean=3.37%, median=1.05%, max=35.10%
- Runs with mean FD > 0.30: 20 / 297
- Runs with >10% volumes FD>0.5: 29 / 297

## Regulation-run motion summary
- Regulation runs found: 66
- Mean FD in regulation runs: mean=0.1663, median=0.1377, max=0.4783
- Regulation runs marked as sensitivity candidates: 29 / 66
- Regulation runs excluded from main: 0 / 66

## High-motion subject candidates
- sub-10: mean FD=0.379, motion-candidate runs=8, reg motion-candidate runs=2
- sub-04: mean FD=0.347, motion-candidate runs=7, reg motion-candidate runs=1
- sub-06: mean FD=0.279, motion-candidate runs=4, reg motion-candidate runs=1
- sub-29: mean FD=0.267, motion-candidate runs=3, reg motion-candidate runs=1
- sub-19: mean FD=0.215, motion-candidate runs=2, reg motion-candidate runs=1
- sub-02: mean FD=0.166, motion-candidate runs=3, reg motion-candidate runs=1
- sub-25: mean FD=0.162, motion-candidate runs=2, reg motion-candidate runs=1

## High nuisance-burden subject candidates
- sub-04: nuisance-candidate runs=4, min estimated DOF=75
- sub-29: nuisance-candidate runs=3, min estimated DOF=91
- sub-25: nuisance-candidate runs=6, min estimated DOF=13
- sub-28: nuisance-candidate runs=5, min estimated DOF=12
- sub-03: nuisance-candidate runs=3, min estimated DOF=70
- sub-23: nuisance-candidate runs=1, min estimated DOF=99
- sub-27: nuisance-candidate runs=5, min estimated DOF=43
- sub-01: nuisance-candidate runs=4, min estimated DOF=44
- sub-13: nuisance-candidate runs=3, min estimated DOF=82
- sub-20: nuisance-candidate runs=2, min estimated DOF=77
- sub-09: nuisance-candidate runs=2, min estimated DOF=94
- sub-11: nuisance-candidate runs=4, min estimated DOF=37
- sub-16: nuisance-candidate runs=5, min estimated DOF=33
- sub-22: nuisance-candidate runs=1, min estimated DOF=77
- sub-26: nuisance-candidate runs=3, min estimated DOF=63
- sub-31: nuisance-candidate runs=4, min estimated DOF=47
- sub-33: nuisance-candidate runs=8, min estimated DOF=21
- sub-30: nuisance-candidate runs=6, min estimated DOF=-1
- sub-24: nuisance-candidate runs=3, min estimated DOF=59
- sub-18: nuisance-candidate runs=5, min estimated DOF=58
- sub-05: nuisance-candidate runs=3, min estimated DOF=88
- sub-32: nuisance-candidate runs=7, min estimated DOF=18
- sub-14: nuisance-candidate runs=4, min estimated DOF=79
- sub-17: nuisance-candidate runs=4, min estimated DOF=10
- sub-21: nuisance-candidate runs=5, min estimated DOF=26
- sub-15: nuisance-candidate runs=2, min estimated DOF=69

## Interpretation policy
- `EXCLUDE` marks severe file or FD-based motion problems; low estimated DOF is retained for main analyses but flagged for design review and sensitivity analyses.
- `KEEP_FOR_MAIN_BUT_SENSITIVITY_EXCLUDE_CANDIDATE` means the run remains eligible for the main analysis and is excluded in the corresponding sensitivity analysis.
- `KEEP_WITH_WARNING` marks mild-to-moderate motion or nuisance burden.
- `motion_outlier*` columns are interpreted as nuisance-regressor burden, not as automatic deletion criteria.

## Key outputs
- `tables/run_qc_fmriprep.tsv`
- `tables/subject_qc_summary.tsv`
- `tables/regulation_run_qc.tsv`
- `tables/high_motion_run_candidates.tsv`
- `tables/high_motion_subject_candidates.tsv`
- `tables/analysis_inclusion_table.tsv`
- `figures/`
