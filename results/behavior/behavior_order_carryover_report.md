# Behavioral order and post-regulation carryover report

## Overview
- Generated: 2026-06-07T04:52:37
- Input: `<PROJECT_ROOT>/derivatives/beh/behavior_trials_master_clean.tsv`
- Subjects: 33
- Total trials: 3201
- Regulation trials: 660
- Post-regulation passive trials: 726

## Why order is tested at subject level
Regulation occurs only in run-03 and run-07. Therefore, condition, run position, and order are structurally coupled. Trial-level condition × order models can be rank-deficient or difficult to interpret. The primary order tests here are subject-level tests on regulation success and asymmetry summaries.

## Order-effect summary
- order_effect_on_mean_success_z: diff(up_first - down_first) = 0.0886, p=0.4276, q=0.6414
- order_effect_on_success_asymmetry_z: diff(up_first - down_first) = 0.2552, p=0.1636, q=0.4907
- order_effect_on_up_success_z: diff(up_first - down_first) = 0.2162, p=0.1459, q=0.4907
- order_effect_on_down_success_z: diff(up_first - down_first) = -0.03902, p=0.7791, q=0.7791

## First-vs-second regulation run
- first minus second regulation success: mean=0.1263, p=0.1514, q=0.1514

## Post-regulation carryover
- post_up_residual_z_gt0: mean=-0.04471, p=0.3635, q=0.727
- post_down_residual_z_gt0: mean=-0.07398, p=0.2364, q=0.7091
- post_up_minus_post_down_residual_z: mean=0.02927, p=0.698, q=0.8376
- run4_minus_run8_post_residual_z: mean=-0.1136, p=0.1252, q=0.7091

## Trial-level carryover models
- post_residual_prev_condition_subjectFE | C(prev_reg_condition)[T.up]: beta=0.02458, p=0.7417, q=0.7417
- post_residual_prev_condition_subjectFE | post_reg_epoch: beta=0.1072, p=0.1627, q=0.3255
- rating_z_prev_condition_subjectFE | C(prev_reg_condition)[T.up]: beta=0.02458, p=0.7417, q=0.7417
- rating_z_prev_condition_subjectFE | post_reg_epoch: beta=-0.06255, p=0.4105, q=0.5473
- post_residual_prev_code_subjectFE | prev_reg_code: beta=0.01229, p=0.7417, q=0.7417
- post_residual_prev_code_subjectFE | post_reg_epoch: beta=0.1072, p=0.1627, q=0.3255

## Interpretation notes
- Order effects are interpreted conservatively because order, condition, and run position are linked by design.
- Carryover into run-04/run-08 is tested using residuals from a passive baseline model fit on non-post-regulation passive runs.
- A positive post-up minus post-down residual means higher-than-expected pain after an up-regulation run relative to after a down-regulation run.
