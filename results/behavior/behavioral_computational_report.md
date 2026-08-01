# ds000140 behavioral computational analysis

## Input and reproducibility
- Generated: 2026-05-07T09:22:02
- Input: `<PROJECT_ROOT>/derivatives/beh/behavior_trials_master_clean.tsv`
- SHA256: `632695cbcdc3cdce0a471699cd35fc793c22ee739b1d81df715960cc016198dd`
- Python: `3.11.14 (main, Oct 21 2025, 18:31:21) [GCC 11.2.0]`

## Dataset summary
- Subjects: 33
- Trials: 3201 total; passive=2541, up=330, down=330
- Regulation runs: 3,7
- Order balance: up-first=16, down-first=17
- rating_raw range: 0.0 to 200.0; values >100: 1699

## 1. Task validation: does explicit regulation change pain reports relative to passive?
- Up vs passive, rating_z: β=0.3402, SE=0.0769, p=0.0001.
- Down vs passive, rating_z: β=-0.2885, SE=0.0703, p=0.0003.
- Temperature effect, rating_z: β=0.6978, SE=0.0203, p=7.67e-27.

## 2. Regulation success and true behavioral asymmetry
- up_success_z_gt_0: mean=0.3334, dz=0.789, p_t=7.74e-05, sign-flip p=1.00e-04.
- down_success_z_gt_0: mean=0.2949, dz=0.760, p_t=0.0001, sign-flip p=5.00e-05.
- mean_regulation_success_z_gt_0: mean=0.3142, dz=0.993, p_t=2.54e-06, sign-flip p=1.00e-04.
- True success asymmetry, success_up − success_down: mean=0.0385, dz=0.076, p_t=0.6668, sign-flip p=0.6610.
- Trial-level success asymmetry model, is_up coefficient: β=0.0423, SE=0.0894, p=0.6393.

Important: up-minus-down residual/rating separation is reported separately and is not labeled as success asymmetry. True asymmetry is tested on direction-corrected success.

## 3–4. Within-run and across-run habituation/sensitization
- Passive within-run rating drift: β=-0.0533, SE=0.0055, p=4.29e-11, q=3.11e-10.
- Within-run regulation-success drift: β=-0.0349, SE=0.0161, p=0.0374, q=0.0598.
- Up/down difference in success drift: β=0.0190, SE=0.0232, p=0.4188, q=0.5584.
- Passive across-run linear drift: β=-0.0266, SE=0.0117, p=0.0298, q=0.0596.
- Passive across-run quadratic drift: β=-0.0010, SE=0.0053, p=0.8530, q=0.8772.

## 5. Regulation order: up-first vs down-first
- order_effect_on_mean_success_z: group difference up_first−down_first=0.0886, p=0.4276, q=0.5131.
- order_effect_on_success_asymmetry_z: group difference up_first−down_first=0.2552, p=0.1636, q=0.3271.
- order_effect_on_task_separation_rating_z: group difference up_first−down_first=0.3402, p=0.1232, q=0.3271.

Order tests are subject-level Welch tests. The problematic trial-level condition × order model is intentionally omitted because condition, run, and order are structurally coupled in this design.

## Computational model comparison
- AIC/BIC comparisons were computed on the same complete-case trial set (n=3168) to avoid unequal-N information-criterion artifacts.
- Best AIC model: M4_state_dependent_regulation (ΔAIC=0, ΔBIC=7.71).
- Best BIC model: M3_history_state (ΔBIC=0, ΔAIC=4.41).
- Nested LRT M0_sensory_only -> M1_additive_regulatory_shift: χ²=142.51, df=2, p=1.13e-31, q=2.26e-31; ΔBIC(full−restricted)=-126.39.
- Nested LRT M1_additive_regulatory_shift -> M2_gain_modulation: χ²=3.80, df=2, p=0.1498, q=0.1498; ΔBIC(full−restricted)=12.33.
- Nested LRT M1_additive_regulatory_shift -> M3_history_state: χ²=194.52, df=2, p=5.75e-43, q=2.30e-42; ΔBIC(full−restricted)=-178.40.
- Nested LRT M3_history_state -> M4_state_dependent_regulation: χ²=8.41, df=2, p=0.0149, q=0.0199; ΔBIC(full−restricted)=7.71.
- LOSO predictive check without subject fixed effects: best RMSE model was M4_state_dependent_regulation (RMSE=0.6535). This is complementary and not used as the primary inferential test.
- These models compare additive regulatory shift, sensory-gain modulation, and history/state terms. Treat model comparison as a behavioral decomposition that generates regressors for later fMRI analyses, not as causal proof by itself.

## Key output files
- `tables/trialwise_behavior_latents.tsv`: expected pain, prediction error, and regulation-success variables for fMRI linkage.
- `tables/subject_level_behavior_parameters.tsv`: subject-level sensitivity/success/asymmetry/order metrics.
- `tables/task_validation_condition_effects.tsv`: regulation vs passive validation.
- `tables/regulation_success_subject_tests.tsv` and `tables/success_asymmetry_subject_tests.tsv`: core subject-level behavioral inference.
- `tables/habituation_time_effects.tsv`: within-run and across-run drift tests.
- `tables/order_effect_subject_tests.tsv`: order analyses.
- `tables/computational_model_comparison.tsv`: common-N AIC/BIC model comparison.
- `tables/computational_model_nested_tests.tsv`: likelihood-ratio tests for nested mechanism models.
- `tables/computational_model_loso_cv.tsv`: leave-one-subject-out predictive sensitivity check.
- `figures/`: QC and summary figures.
