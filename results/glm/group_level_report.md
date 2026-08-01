# Group-level GLM report

## Overview
- Generated: 2026-05-08T09:23:55
- GLM directory: `<PROJECT_ROOT>/derivatives/firstlevel_GLM_analysis`
- Sensitivity table: `<PROJECT_ROOT>/derivatives/first_level_inputs_analysis/tables/sensitivity_sets.tsv`
- Output directory: `<PROJECT_ROOT>/derivatives/group_level_analysis`
- Sets: main, fd_bad_run_exclude, high_motion_subject_exclude, motion_covariate_mean_fd
- Effects: up_minus_down_FE, up_minus_passive_FE, down_minus_passive_FE, regulation_mean_minus_passive_FE, stim_passive_FE

## Robustness sets
- `main`: all runs passing severe QC and first-level GLM.
- `fd_bad_run_exclude`: rebuild subject effects excluding only FD-based bad/severe runs.
- `high_motion_subject_exclude`: use main subject effects but exclude high-motion subjects.
- `motion_covariate_mean_fd`: use main subject effects and add subject mean FD as a second-level covariate.
- `strict`: optional stress test excluding all sensitivity candidates; can be underpowered for regulation contrasts.

## High-motion subjects
- High-motion subjects excluded in high_motion_subject_exclude: 7
- Subjects: sub-02, sub-04, sub-06, sub-10, sub-19, sub-25, sub-29

## Subject fixed effects
- Subject-effect rows: 882
- fd_bad_run_exclude: 33 subjects with at least one effect row
- high_motion_subject_exclude: 26 subjects with at least one effect row
- main: 33 subjects with at least one effect row

## Group-level results
- main | up_minus_down_FE: status=OK, n=33, zmaxabs=2.476
- main | up_minus_passive_FE: status=OK, n=33, zmaxabs=2.702
- main | down_minus_passive_FE: status=OK, n=33, zmaxabs=4.508
- main | regulation_mean_minus_passive_FE: status=OK, n=33, zmaxabs=2.316
- main | stim_passive_FE: status=OK, n=33, zmaxabs=6.205
- fd_bad_run_exclude | up_minus_down_FE: status=OK, n=26, zmaxabs=4.091
- fd_bad_run_exclude | up_minus_passive_FE: status=OK, n=28, zmaxabs=3.037
- fd_bad_run_exclude | down_minus_passive_FE: status=OK, n=30, zmaxabs=4.818
- fd_bad_run_exclude | regulation_mean_minus_passive_FE: status=OK, n=26, zmaxabs=4.197
- fd_bad_run_exclude | stim_passive_FE: status=OK, n=33, zmaxabs=6.159
- high_motion_subject_exclude | up_minus_down_FE: status=OK, n=26, zmaxabs=4.091
- high_motion_subject_exclude | up_minus_passive_FE: status=OK, n=26, zmaxabs=4.170
- high_motion_subject_exclude | down_minus_passive_FE: status=OK, n=26, zmaxabs=4.564
- high_motion_subject_exclude | regulation_mean_minus_passive_FE: status=OK, n=26, zmaxabs=4.181
- high_motion_subject_exclude | stim_passive_FE: status=OK, n=26, zmaxabs=5.972
- motion_covariate_mean_fd | up_minus_down_FE: status=OK, n=33, zmaxabs=3.092
- motion_covariate_mean_fd | up_minus_passive_FE: status=OK, n=33, zmaxabs=2.933
- motion_covariate_mean_fd | down_minus_passive_FE: status=OK, n=33, zmaxabs=4.475
- motion_covariate_mean_fd | regulation_mean_minus_passive_FE: status=OK, n=33, zmaxabs=2.638
- motion_covariate_mean_fd | stim_passive_FE: status=OK, n=33, zmaxabs=6.126

## Robustness summary versus main
- up_minus_down_FE | fd_bad_run_exclude: n=26, beta corr=0.002, z corr=-0.015
- up_minus_down_FE | high_motion_subject_exclude: n=26, beta corr=0.002, z corr=-0.015
- up_minus_down_FE | motion_covariate_mean_fd: n=33, beta corr=1.000, z corr=1.000
- up_minus_passive_FE | fd_bad_run_exclude: n=28, beta corr=1.000, z corr=0.969
- up_minus_passive_FE | high_motion_subject_exclude: n=26, beta corr=0.002, z corr=0.131
- up_minus_passive_FE | motion_covariate_mean_fd: n=33, beta corr=1.000, z corr=1.000
- down_minus_passive_FE | fd_bad_run_exclude: n=30, beta corr=0.892, z corr=0.923
- down_minus_passive_FE | high_motion_subject_exclude: n=26, beta corr=0.844, z corr=0.874
- down_minus_passive_FE | motion_covariate_mean_fd: n=33, beta corr=1.000, z corr=1.000
- regulation_mean_minus_passive_FE | fd_bad_run_exclude: n=26, beta corr=0.226, z corr=0.191
- regulation_mean_minus_passive_FE | high_motion_subject_exclude: n=26, beta corr=0.230, z corr=0.175
- regulation_mean_minus_passive_FE | motion_covariate_mean_fd: n=33, beta corr=1.000, z corr=1.000
- stim_passive_FE | fd_bad_run_exclude: n=33, beta corr=0.985, z corr=0.985
- stim_passive_FE | high_motion_subject_exclude: n=26, beta corr=0.966, z corr=0.955
- stim_passive_FE | motion_covariate_mean_fd: n=33, beta corr=1.000, z corr=1.000

## Interpretation policy
- The `main` analysis set defines primary inference.
- Robustness is evaluated by similarity across FD-bad-run exclusion, high-motion-participant exclusion, and motion-covariate control.
- The strict set is intentionally conservative and may be underpowered for regulation contrasts due to the two-run regulation design.
- Inference is based on unthresholded group maps, corrected/thresholded maps, and consistency across robustness analyses.

