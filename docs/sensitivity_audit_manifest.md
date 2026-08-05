# Sensitivity-analysis map

| Question | Code | Included outputs |
|---|---|---|
| Do behavioral conclusions survive finite-cluster inference? | `code/07_sensitivity_audits/run_behavior_small_sample_inference.R` | `results/sensitivity_audits/behavior_small_sample_inference/` |
| Does the selected coupling survive AR(1) prewhitening? | `code/07_sensitivity_audits/run_ar1_prewhitening_sensitivity.py`; `run_matched_coupling_controls.py` | `results/sensitivity_audits/ar1_prewhitening_roi/`; `matched_coupling_controls_ar1/` |
| Is the state-dependence interaction robust to scale and change-score coupling? | `code/07_sensitivity_audits/run_state_dependence_coupling_null_controls.py` | `results/sensitivity_audits/state_dependence_coupling_null_controls/` |
| Are down-minus-passive ROI effects stable across analysis sets? | `code/07_sensitivity_audits/run_down_regulation_attenuation_robustness.py` | `results/sensitivity_audits/down_regulation_attenuation_robustness/` |
| Do matched passive runs alter regional inferences? | `code/07_sensitivity_audits/run_neural_matched_passive_roi.py` | `results/sensitivity_audits/neural_matched_passive_roi/` |
| Is NPS inference consistent across contrast constructions and participants? | `code/07_sensitivity_audits/run_nps_contrast_consistency_audit.py`; `run_nps_participant_influence_audit.py` | `results/sensitivity_audits/nps_contrast_consistency/`; `nps_participant_influence/` |
| Does the selected coupling survive matched temporal and conditional null controls? | `code/07_sensitivity_audits/run_matched_coupling_controls.py` | `results/sensitivity_audits/matched_coupling_controls/` |
| Does run-separated DCM independently support the coupling interpretation? | `code/06_dcm/` | `results/reporting/table_data/supplementary_table_s06_source_data.tsv` |

All coupling follow-ups are conditional on the selected pathway and do not repeat the initial screening family.
