# Reduced pathway follow-up models report

## Purpose
This analysis applies a reduced post-selection set of pathway-behavior models after the targeted screening. It evaluates whether individual differences in the selected lagged control-value composite are associated with regulation success after covariate adjustment; it is not an independent confirmation.

## Inputs
- Pathway scores: `<PROJECT_ROOT>/derivatives/targeted_pathway_FC_EC_behavior_analysis/tables/pathway_scores.tsv`
- Behavior subjects: `<PROJECT_ROOT>/derivatives/beh/behavior_computational_analysis/tables/subject_level_behavior_parameters.tsv`
- Output directory: `<PROJECT_ROOT>/derivatives/screening_reduced_pathway_models`
- Default covariates: `up_first, passive_mean_rating_z, passive_sensory_slope_z`
- Bootstrap samples: 5000
- Permutations: 10000

## EC interpretation warning
EC-lite is directed lagged predictive connectivity, not DCM. It is treated as an exploratory temporal-prediction metric and does not establish causal or physiological directionality.

## Family-level summary
- primary | primary_control_value_EC_behavior: primary_ec_dlpfc_vmPFC_ec_up_to_reg_success_z_up | beta=-0.4822, p=0.001354, q=0.002707, partial_r=-0.5829, perm_p=0.0003, bootstrap CI=[-0.6936, -0.2192], LOO same sign=100.0%
- secondary | secondary_control_activity_behavior: secondary_roi_control_dlPFC_IFJ_L_regulation_mean_minus_passive_to_reg_success_z_up | beta=0.161, p=0.3819, q=0.8435, partial_r=0.2001, perm_p=0.2754, bootstrap CI=[-0.1441, 0.4705], LOO same sign=100.0%
- secondary | secondary_sensory_FC_behavior: secondary_fc_sensory_thalamus_insula_down_to_reg_success_z_down | beta=-0.05064, p=0.8479, q=0.8479, partial_r=-0.05972, perm_p=0.7639, bootstrap CI=[-0.4366, 0.34], LOO same sign=90.9%

## Primary models
- primary_ec_dlpfc_vmPFC_ec_up_to_reg_success_z_up: outcome=reg_success_z_up, predictor=ec_up (dlPFC_IFJ_L_to_vmPFC_control_value); beta=-0.4822, p=0.001354, q=0.002707, partial_r=-0.5829, perm_p=0.0003
- primary_ec_dlpfc_vmPFC_ec_up_to_reg_success_raw_up: outcome=reg_success_raw_up, predictor=ec_up (dlPFC_IFJ_L_to_vmPFC_control_value); beta=-0.5358, p=0.0004126, q=0.002707, partial_r=-0.6274, perm_p=0.0002
- primary_ec_dlpfc_vmPFC_ec_up_to_mean_reg_success_z: outcome=mean_reg_success_z, predictor=ec_up (dlPFC_IFJ_L_to_vmPFC_control_value); beta=-0.65, p=0.001207, q=0.002707, partial_r=-0.5902, perm_p=0.0007999
- primary_ec_dlpfc_vmPFC_ec_up_to_mean_reg_success_raw: outcome=mean_reg_success_raw, predictor=ec_up (dlPFC_IFJ_L_to_vmPFC_control_value); beta=-0.7143, p=0.0007109, q=0.002707, partial_r=-0.652, perm_p=0.0003
- primary_ec_dlpfc_vmPFC_up_minus_passive_to_reg_success_raw_up: outcome=reg_success_raw_up, predictor=up_minus_passive (dlPFC_IFJ_L_to_vmPFC_control_value); beta=-0.4633, p=0.001737, q=0.002779, partial_r=-0.5594, perm_p=0.0008999
- primary_ec_dlpfc_vmPFC_up_minus_passive_to_mean_reg_success_raw: outcome=mean_reg_success_raw, predictor=up_minus_passive (dlPFC_IFJ_L_to_vmPFC_control_value); beta=-0.6225, p=0.003007, q=0.00401, partial_r=-0.5859, perm_p=0.0011
- primary_ec_dlpfc_vmPFC_up_minus_passive_to_reg_success_z_up: outcome=reg_success_z_up, predictor=up_minus_passive (dlPFC_IFJ_L_to_vmPFC_control_value); beta=-0.4096, p=0.007968, q=0.007968, partial_r=-0.5105, perm_p=0.003
- primary_ec_dlpfc_vmPFC_up_minus_passive_to_mean_reg_success_z: outcome=mean_reg_success_z, predictor=up_minus_passive (dlPFC_IFJ_L_to_vmPFC_control_value); beta=-0.5639, p=0.007031, q=0.007968, partial_r=-0.528, perm_p=0.0023

## Interpretation guide
Concordance among the HC3 p-value, partial correlation, permutation p-value, bootstrap CI, and leave-one-participant-out sign stability strengthens the descriptive evidence. Attenuation after covariate adjustment reinforces the exploratory interpretation.
