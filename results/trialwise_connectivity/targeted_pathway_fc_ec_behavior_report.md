# Targeted pathway screening report

## Purpose
This analysis evaluates a prespecified set of regulatory, value, salience/interoceptive, sensory, and descending-modulation pathways. It combines targeted ROI activity, beta-series functional connectivity, lagged predictive connectivity, and brain-behavior associations.

## Inputs
- ROI activity summary: `results/trialwise_connectivity/roi_activity_subject_summary.tsv`
- FC edges: `results/trialwise_connectivity/fc_beta_series_edges.tsv.gz`
- EC edges: `results/trialwise_connectivity/ec_lagged_edges.tsv.gz`
- Behavior subjects: `results/behavior/subject_level_behavior_parameters.tsv`
- Output tables: `results/trialwise_connectivity/`

## Interpretation of lagged predictive connectivity
Lagged predictive-connectivity results are regression-based temporal association estimates, not DCM. They are interpreted as exploratory pathway-prioritization measures and do not establish causal direction.

## Top targeted ROI activity tests
- sensory_S1_S2_thalamus | sensory_S2_operculum_R; quantity=beta_passive; mean/r=825.3; p=1.172e-12; q=2.343e-11; n=33
- salience_aINS | reg_anterior_insula_L; quantity=beta_passive; mean/r=549.6; p=9.36e-12; q=9.36e-11; n=33
- sensory_S1_S2_thalamus | sensory_S1_L; quantity=beta_passive; mean/r=-672.8; p=2.112e-10; q=1.408e-09; n=33
- sensory_S1_S2_thalamus | sensory_S2_operculum_R; quantity=beta_up; mean/r=935.5; p=1.173e-09; q=2.345e-08; n=33
- sensory_S1_S2_thalamus | sensory_S2_operculum_R; quantity=beta_down; mean/r=779.7; p=2.54e-09; q=5.08e-08; n=33
- salience_aINS | reg_anterior_insula_R; quantity=beta_passive; mean/r=690.9; p=1.23e-08; q=6.149e-08; n=33
- sensory_S1_S2_thalamus | sensory_S1_L; quantity=beta_down; mean/r=-766; p=1.806e-08; q=1.806e-07; n=33
- sensory_S1_S2_thalamus | sensory_thalamus_L; quantity=beta_passive; mean/r=-208; p=3.492e-06; q=1.397e-05; n=33
- salience_aINS | reg_anterior_insula_R; quantity=beta_up; mean/r=736.2; p=7.242e-06; q=7.242e-05; n=33
- salience_aINS | reg_anterior_insula_L; quantity=beta_up; mean/r=500.8; p=1.236e-05; q=8.243e-05; n=33
- sensory_S1_S2_thalamus | sensory_S1_L; quantity=beta_up; mean/r=-582.2; p=3.12e-05; q=0.000156; n=33
- salience_aINS | reg_anterior_insula_L; quantity=beta_down; mean/r=489.8; p=3.072e-05; q=0.0002048; n=33
- descending_pgACC_rACC | reg_pgACC_rACC; quantity=beta_passive; mean/r=125.5; p=6.762e-05; q=0.0002254; n=33
- sensory_S1_S2_thalamus | sensory_thalamus_L; quantity=beta_down; mean/r=-216.5; p=0.0004059; q=0.00203; n=33
- sensory_S1_S2_thalamus | sensory_S1_R; quantity=beta_passive; mean/r=-169.9; p=0.001251; q=0.003576; n=33
- descending_pgACC_rACC | reg_pgACC_rACC; quantity=beta_down; mean/r=204.5; p=0.001188; q=0.00475; n=33
- value_NAc | reg_NAc_L; quantity=beta_passive; mean/r=125.2; p=0.001918; q=0.004795; n=33
- value_NAc | reg_NAc_R; quantity=beta_down; mean/r=294.9; p=0.001532; q=0.005105; n=33
- sensory_S1_S2_thalamus | sensory_S2_operculum_L; quantity=beta_passive; mean/r=266.8; p=0.002815; q=0.006255; n=33
- value_NAc | reg_NAc_R; quantity=beta_passive; mean/r=119.1; p=0.003798; q=0.007595; n=33

## Top targeted FC tests
- aINS_dACC_MCC_salience | sensory_dACC_MCC | sensory_insula_all_L; quantity=z_passive; mean/r=0.4533; p=1.43e-13; q=6.578e-12; n=33
- aINS_dACC_MCC_salience | reg_anterior_insula_L | sensory_dACC_MCC; quantity=z_passive; mean/r=0.3466; p=3.836e-12; q=8.823e-11; n=33
- aINS_dACC_MCC_salience | sensory_dACC_MCC | sensory_insula_all_R; quantity=z_passive; mean/r=0.4147; p=7.062e-12; q=1.083e-10; n=33
- aINS_dACC_MCC_salience | sensory_dACC_MCC | sensory_posterior_insula_L; quantity=z_passive; mean/r=0.3557; p=4.73e-11; q=5.44e-10; n=33
- sensory_thalamus_S1_S2_insula | sensory_insula_all_R | sensory_thalamus_R; quantity=z_passive; mean/r=0.2435; p=4.751e-09; q=4.371e-08; n=33
- aINS_dACC_MCC_salience | reg_anterior_insula_R | sensory_dACC_MCC; quantity=z_passive; mean/r=0.2601; p=1.309e-08; q=1.003e-07; n=33
- sensory_thalamus_S1_S2_insula | reg_anterior_insula_R | sensory_thalamus_R; quantity=z_passive; mean/r=0.2204; p=1.856e-08; q=1.22e-07; n=33
- aINS_dACC_MCC_salience | sensory_dACC_MCC | sensory_posterior_insula_R; quantity=z_passive; mean/r=0.3135; p=5.646e-08; q=3.247e-07; n=33
- NAc_vmPFC_value | reg_NAc_L | reg_vmPFC_mOFC; quantity=z_passive; mean/r=0.1651; p=8.181e-08; q=4.182e-07; n=33
- sensory_thalamus_S1_S2_insula | reg_anterior_insula_L | sensory_thalamus_R; quantity=z_passive; mean/r=0.1848; p=6.503e-07; q=2.991e-06; n=33
- NAc_vmPFC_value | reg_NAc_L | reg_vmPFC; quantity=z_passive; mean/r=0.1549; p=1.564e-06; q=6.538e-06; n=33
- sensory_thalamus_S1_S2_insula | sensory_insula_all_L | sensory_thalamus_R; quantity=z_passive; mean/r=0.2069; p=1.836e-06; q=7.037e-06; n=33
- sensory_thalamus_S1_S2_insula | sensory_insula_all_L | sensory_thalamus_L; quantity=z_passive; mean/r=0.2069; p=4.576e-06; q=1.619e-05; n=33
- sensory_thalamus_S1_S2_insula | sensory_posterior_insula_L | sensory_thalamus_R; quantity=z_passive; mean/r=0.1712; p=6.607e-06; q=2.081e-05; n=33
- sensory_thalamus_S1_S2_insula | reg_anterior_insula_L | sensory_thalamus_L; quantity=z_passive; mean/r=0.1904; p=6.787e-06; q=2.081e-05; n=33
- sensory_thalamus_S1_S2_insula | sensory_S2_operculum_R | sensory_thalamus_R; quantity=z_passive; mean/r=0.158; p=1.594e-05; q=4.582e-05; n=33
- aINS_dACC_MCC_salience | sensory_dACC_MCC | sensory_insula_all_L; quantity=z_down; mean/r=0.5076; p=1.781e-06; q=8.195e-05; n=33
- sensory_thalamus_S1_S2_insula | sensory_posterior_insula_L | sensory_thalamus_L; quantity=z_passive; mean/r=0.169; p=4.061e-05; q=0.0001099; n=33
- aINS_dACC_MCC_salience | sensory_dACC_MCC | sensory_insula_all_R; quantity=z_down; mean/r=0.4344; p=5.931e-06; q=0.0001364; n=33
- aINS_dACC_MCC_salience | sensory_dACC_MCC | sensory_insula_all_R; quantity=z_up; mean/r=0.4392; p=3.744e-06; q=0.0001722; n=33

## Top targeted EC-lite tests
- aINS_to_dACC_MCC_salience | reg_anterior_insula_R | sensory_dACC_MCC; quantity=ec_up; mean/r=0.1937; p=0.01343; q=0.2148; n=33
- aINS_to_dACC_MCC_salience | reg_anterior_insula_R | sensory_dACC_MCC; quantity=up_minus_passive; mean/r=0.1825; p=0.01963; q=0.3141; n=33
- NAc_to_vmPFC_value | reg_NAc_L | reg_mOFC; quantity=ec_passive; mean/r=0.05404; p=0.02973; q=0.3689; n=33
- aINS_to_dACC_MCC_salience | sensory_posterior_insula_L | sensory_dACC_MCC; quantity=ec_passive; mean/r=-0.04442; p=0.04611; q=0.3689; n=33
- aINS_to_dACC_MCC_salience | reg_anterior_insula_R | sensory_dACC_MCC; quantity=up_minus_down; mean/r=0.2461; p=0.02377; q=0.3803; n=33
- aINS_to_dACC_MCC_salience | reg_anterior_insula_L | sensory_dACC_MCC; quantity=regulation_mean_minus_passive; mean/r=0.03862; p=0.513; q=0.7459; n=33
- NAc_to_vmPFC_value | reg_NAc_L | reg_vmPFC_mOFC; quantity=regulation_mean_minus_passive; mean/r=-0.05405; p=0.4257; q=0.7459; n=33
- NAc_to_vmPFC_value | reg_NAc_L | reg_mOFC; quantity=regulation_mean_minus_passive; mean/r=-0.0524; p=0.4098; q=0.7459; n=33
- aINS_to_dACC_MCC_salience | sensory_insula_all_L | sensory_dACC_MCC; quantity=regulation_mean_minus_passive; mean/r=0.04059; p=0.4983; q=0.7459; n=33
- pgACC_rACC_to_PAG_descending | reg_pgACC_rACC | reg_PAG_sphere; quantity=regulation_mean_minus_passive; mean/r=-0.05491; p=0.2603; q=0.7459; n=33
- dlPFC_IFJ_L_to_vmPFC_control_value | reg_dlPFC_IFJ_L | reg_vmPFC; quantity=regulation_mean_minus_passive; mean/r=-0.06595; p=0.3031; q=0.7459; n=33
- dlPFC_IFJ_L_to_vmPFC_control_value | reg_dlPFC_IFJ_L | reg_mOFC; quantity=regulation_mean_minus_passive; mean/r=-0.04023; p=0.5594; q=0.7459; n=33
- aINS_to_dACC_MCC_salience | sensory_posterior_insula_R | sensory_dACC_MCC; quantity=regulation_mean_minus_passive; mean/r=0.06766; p=0.3808; q=0.7459; n=33
- aINS_to_dACC_MCC_salience | sensory_posterior_insula_L | sensory_dACC_MCC; quantity=regulation_mean_minus_passive; mean/r=0.0751; p=0.2672; q=0.7459; n=33
- aINS_to_dACC_MCC_salience | sensory_insula_all_R | sensory_dACC_MCC; quantity=regulation_mean_minus_passive; mean/r=0.06832; p=0.289; q=0.7459; n=33
- aINS_to_dACC_MCC_salience | reg_anterior_insula_R | sensory_dACC_MCC; quantity=regulation_mean_minus_passive; mean/r=0.05942; p=0.3187; q=0.7459; n=33
- dlPFC_IFJ_L_to_vmPFC_control_value | reg_dlPFC_IFJ_L | reg_vmPFC_mOFC; quantity=regulation_mean_minus_passive; mean/r=-0.04544; p=0.4778; q=0.7459; n=33
- aINS_to_dACC_MCC_salience | sensory_insula_all_R | sensory_dACC_MCC; quantity=up_minus_down; mean/r=0.09208; p=0.4377; q=0.8384; n=33
- dlPFC_IFJ_L_to_vmPFC_control_value | reg_dlPFC_IFJ_L | reg_vmPFC_mOFC; quantity=up_minus_down; mean/r=0.03608; p=0.7574; q=0.8384; n=33
- dlPFC_IFJ_L_to_vmPFC_control_value | reg_dlPFC_IFJ_L | reg_vmPFC; quantity=up_minus_down; mean/r=0.09574; p=0.3839; q=0.8384; n=33

## Top pathway-score tests
- FC_beta_series | aINS_dACC_MCC_salience; quantity=z_passive; mean/r=0.3573; p=1.081e-13; q=5.407e-13; n=33
- ROI_activity | salience_aINS; quantity=beta_passive; mean/r=620.3; p=9.619e-11; q=9.619e-10; n=33
- ROI_activity | salience_aINS; quantity=beta_up; mean/r=618.5; p=2.177e-07; q=2.177e-06; n=33
- FC_beta_series | NAc_vmPFC_value; quantity=z_passive; mean/r=0.1333; p=1.057e-06; q=2.643e-06; n=33
- FC_beta_series | aINS_dACC_MCC_salience; quantity=z_down; mean/r=0.4076; p=9.319e-07; q=4.659e-06; n=33
- FC_beta_series | aINS_dACC_MCC_salience; quantity=z_up; mean/r=0.3408; p=2.1e-06; q=1.05e-05; n=33
- FC_beta_series | NAc_vmPFC_value; quantity=z_up; mean/r=0.2814; p=1.606e-05; q=4.015e-05; n=33
- FC_beta_series | sensory_thalamus_S1_S2_insula; quantity=z_passive; mean/r=0.09071; p=4.404e-05; q=7.34e-05; n=33
- ROI_activity | descending_pgACC_rACC; quantity=beta_passive; mean/r=125.5; p=6.762e-05; q=0.0003381; n=33
- ROI_activity | salience_aINS; quantity=beta_down; mean/r=565.9; p=0.0001915; q=0.001915; n=33
- FC_beta_series | dlPFC_IFJ_L_vmPFC_control_value; quantity=z_passive; mean/r=0.1215; p=0.001955; q=0.002443; n=33
- ROI_activity | value_NAc; quantity=beta_passive; mean/r=122.1; p=0.001024; q=0.003413; n=33
- ROI_activity | descending_pgACC_rACC; quantity=beta_down; mean/r=204.5; p=0.001188; q=0.005938; n=33
- ROI_activity | value_NAc; quantity=beta_down; mean/r=260.2; p=0.00207; q=0.0069; n=33
- ROI_activity | control_SMA_preSMA; quantity=regulation_mean_minus_passive; mean/r=197.5; p=0.001332; q=0.01328; n=33
- ROI_activity | control_dlPFC_IFJ_L; quantity=regulation_mean_minus_passive; mean/r=237.2; p=0.002655; q=0.01328; n=33
- FC_beta_series | sensory_thalamus_S1_S2_insula; quantity=z_up; mean/r=0.08199; p=0.01407; q=0.02345; n=33
- ROI_activity | control_SMA_preSMA; quantity=beta_passive; mean/r=-125.2; p=0.0114; q=0.02532; n=33
- ROI_activity | value_vmPFC_mOFC; quantity=beta_passive; mean/r=97.95; p=0.01266; q=0.02532; n=33
- ROI_activity | control_SMA_preSMA; quantity=up_minus_passive; mean/r=225.7; p=0.00801; q=0.04005; n=33

## Top pathway-behavior correlations
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | reg_success_raw_up; quantity=ec_up; mean/r=-0.6101; p=0.0001632; q=0.00631; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | prediction_error_raw_up; quantity=ec_up; mean/r=-0.6101; p=0.0001632; q=0.00631; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | mean_reg_success_raw; quantity=ec_up; mean/r=-0.6213; p=0.0001142; q=0.00631; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | task_separation_rating_raw_up_minus_down; quantity=ec_up; mean/r=-0.5835; p=0.0003653; q=0.01059; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | prediction_error_z_up; quantity=ec_up; mean/r=-0.5626; p=0.0006549; q=0.01266; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | reg_success_z_up; quantity=ec_up; mean/r=-0.5626; p=0.0006549; q=0.01266; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | rating_z_subject_up; quantity=ec_up; mean/r=-0.544; p=0.001067; q=0.01768; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | task_separation_prediction_error_z_up_minus_down; quantity=ec_up; mean/r=-0.5264; p=0.001649; q=0.02126; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | mean_reg_success_z; quantity=ec_up; mean/r=-0.5264; p=0.001649; q=0.02126; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | reg_success_raw_up; quantity=up_minus_passive; mean/r=-0.5549; p=0.0008031; q=0.03105; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | prediction_error_raw_up; quantity=up_minus_passive; mean/r=-0.5549; p=0.0008031; q=0.03105; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | mean_reg_success_raw; quantity=up_minus_passive; mean/r=-0.5621; p=0.0006639; q=0.03105; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | reg_success_z_up; quantity=up_minus_passive; mean/r=-0.512; p=0.002321; q=0.04488; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | task_separation_rating_raw_up_minus_down; quantity=up_minus_passive; mean/r=-0.5211; p=0.001876; q=0.04488; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | prediction_error_z_up; quantity=up_minus_passive; mean/r=-0.512; p=0.002321; q=0.04488; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | rating_z_subject_up; quantity=up_minus_passive; mean/r=-0.495; p=0.003404; q=0.05642; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | mean_reg_success_z; quantity=up_minus_passive; mean/r=-0.4811; p=0.00459; q=0.05719; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | task_separation_prediction_error_z_up_minus_down; quantity=up_minus_passive; mean/r=-0.4811; p=0.00459; q=0.05719; n=33
- EC_lite_directed_lagged | aINS_to_dACC_MCC_salience | up_first; quantity=up_minus_passive; mean/r=0.4777; p=0.00493; q=0.05719; n=33
- EC_lite_directed_lagged | dlPFC_IFJ_L_to_vmPFC_control_value | task_separation_rating_z_up_minus_down; quantity=ec_up; mean/r=-0.4754; p=0.005179; q=0.06008; n=33

## Family-level summary
- salience_interoception: pathway_score_tests; best=aINS_dACC_MCC_salience; quantity=z_passive; p=1.081e-13; q=5.407e-13
- salience_interoception: targeted_fc_tests; best=aINS_dACC_MCC_salience; quantity=z_passive; p=1.43e-13; q=6.578e-12
- sensory: targeted_roi_activity_tests; best=sensory_S2_operculum_R; quantity=beta_passive; p=1.172e-12; q=2.343e-11
- salience: targeted_roi_activity_tests; best=reg_anterior_insula_L; quantity=beta_passive; p=9.36e-12; q=9.36e-11
- salience: pathway_score_tests; best=salience_aINS; quantity=beta_passive; p=9.619e-11; q=9.619e-10
- sensory_interoceptive: targeted_fc_tests; best=sensory_thalamus_S1_S2_insula; quantity=z_passive; p=4.751e-09; q=4.371e-08
- value_regulation: targeted_fc_tests; best=NAc_vmPFC_value; quantity=z_passive; p=8.181e-08; q=4.182e-07
- value_regulation: pathway_score_tests; best=NAc_vmPFC_value; quantity=z_passive; p=1.057e-06; q=2.643e-06
- sensory_interoceptive: pathway_score_tests; best=sensory_thalamus_S1_S2_insula; quantity=z_passive; p=4.404e-05; q=7.34e-05
- descending: targeted_roi_activity_tests; best=reg_pgACC_rACC; quantity=beta_passive; p=6.762e-05; q=0.0002254
- descending: pathway_score_tests; best=descending_pgACC_rACC; quantity=beta_passive; p=6.762e-05; q=0.0003381
- control_value: targeted_fc_tests; best=dlPFC_IFJ_L_vmPFC_control_value; quantity=z_passive; p=0.001095; q=0.00229
- control_value: pathway_score_tests; best=dlPFC_IFJ_L_vmPFC_control_value; quantity=z_passive; p=0.001955; q=0.002443
- value: pathway_score_tests; best=value_NAc; quantity=beta_passive; p=0.001024; q=0.003413
- value: targeted_roi_activity_tests; best=reg_NAc_L; quantity=beta_passive; p=0.001918; q=0.004795
- control_value: pathway_behavior_correlations; best=dlPFC_IFJ_L_to_vmPFC_control_value ~ mean_reg_success_raw; quantity=ec_up; p=0.0001142; q=0.00631
- control: pathway_score_tests; best=control_SMA_preSMA; quantity=regulation_mean_minus_passive; p=0.001332; q=0.01328
- control: targeted_roi_activity_tests; best=reg_SMA_preSMA; quantity=beta_passive; p=0.0114; q=0.0179
- salience_interoception: pathway_behavior_correlations; best=aINS_to_dACC_MCC_salience ~ up_first; quantity=up_minus_passive; p=0.00493; q=0.05719
- value_regulation: pathway_behavior_correlations; best=NAc_to_vmPFC_value ~ passive_mean_rating_raw; quantity=regulation_mean_minus_passive; p=0.001695; q=0.0983
- sensory: pathway_behavior_correlations; best=sensory_S1_S2_thalamus ~ up_first; quantity=up_minus_down; p=0.003387; q=0.1304
- salience: pathway_behavior_correlations; best=salience_aINS ~ task_separation_rating_z_up_minus_down; quantity=up_minus_down; p=0.002311; q=0.1304
- value: pathway_behavior_correlations; best=value_NAc ~ rating_z_subject_up; quantity=up_minus_down; p=0.001797; q=0.1304
- descending: pathway_behavior_correlations; best=descending_pgACC_rACC ~ passive_mean_rating_raw; quantity=down_minus_passive; p=0.008882; q=0.2146
- salience_interoception: targeted_ec_tests; best=aINS_to_dACC_MCC_salience; quantity=ec_up; p=0.01343; q=0.2148
- descending_modulation: pathway_behavior_correlations; best=pgACC_rACC_to_PAG_descending ~ prediction_error_raw_down; quantity=down_minus_passive; p=0.005359; q=0.2181
- descending_modulation: pathway_score_tests; best=pgACC_rACC_PAG_descending; quantity=z_up; p=0.2268; q=0.2268
- control: pathway_behavior_correlations; best=control_dlPFC_IFJ_L ~ passive_mean_rating_raw; quantity=down_minus_passive; p=0.01036; q=0.2312
- sensory: pathway_score_tests; best=sensory_S1_S2_thalamus; quantity=down_minus_passive; p=0.05057; q=0.2529
- descending_modulation: targeted_fc_tests; best=pgACC_rACC_PAG_descending; quantity=z_up; p=0.2268; q=0.3477

## Scope of downstream modeling
The screening results define a restricted set of candidate pathways for downstream sensitivity analyses. Selection from the same dataset remains exploratory and post-selection.
