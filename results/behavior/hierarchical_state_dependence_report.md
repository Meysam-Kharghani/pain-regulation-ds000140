# Hierarchical state-dependence model - analysis

## Aim
This analysis tests state-dependent pain regulation at the trial level using passive-expected pain and subject-level clustering/hierarchical structure.

## Data
- Regulation trials: 660
- Subjects: 33

## Main cluster-robust subject-FE results
| model                     | term                    |        beta |        se |     t_or_z |            p |   n |        r2 |        q_fdr |
|:--------------------------|:------------------------|------------:|----------:|-----------:|-------------:|----:|----------:|-------------:|
| cluster_OLS_subject_FE    | is_up_c                 |  0.0714108  | 0.0821299 |  0.869486  | 0.384581     | 660 | 0.212401  | 0.528799     |
| cluster_OLS_subject_FE    | expected_pain_c         | -0.0153987  | 0.0311547 | -0.494265  | 0.621119     | 660 | 0.212401  | 0.759145     |
| cluster_OLS_subject_FE    | is_up_c:expected_pain_c | -0.199579   | 0.0675958 | -2.95253   | 0.00315179   | 660 | 0.212401  | 0.0145724    |
| cluster_OLS_subject_FE    | global_trial_c          | -0.0693457  | 0.0418332 | -1.65767   | 0.0973837    | 660 | 0.212401  | 0.218929     |
| cluster_OLS_subject_FE    | up_first_c              |  0.443189   | 0.0170491 | 25.9949    | 5.65002e-149 | 660 | 0.212401  | 6.21502e-148 |
| cluster_OLS_no_subject_FE | is_up_c                 |  0.0711144  | 0.0802339 |  0.886338  | 0.375435     | 660 | 0.0425166 | 0.528799     |
| cluster_OLS_no_subject_FE | expected_pain_c         | -0.00443287 | 0.0750659 | -0.0590531 | 0.95291      | 660 | 0.0425166 | 0.95291      |
| cluster_OLS_no_subject_FE | is_up_c:expected_pain_c | -0.199362   | 0.0692183 | -2.88019   | 0.0039743    | 660 | 0.0425166 | 0.0145724    |
| cluster_OLS_no_subject_FE | global_trial_c          | -0.068182   | 0.0413922 | -1.64722   | 0.0995134    | 660 | 0.0425166 | 0.218929     |
| cluster_OLS_no_subject_FE | up_first_c              |  0.0998258  | 0.0953484 |  1.04696   | 0.295119     | 660 | 0.0425166 | 0.528799     |
| cluster_OLS_no_subject_FE | temperature_c           | -0.0159509  | 0.0656005 | -0.243153  | 0.807887     | 660 | 0.0425166 | 0.888676     |

## Derived up/down state-dependence contrasts
| contrast                       |   estimate |        se |         z |          p | model                  |     q_fdr |
|:-------------------------------|-----------:|----------:|----------:|-----------:|:-----------------------|----------:|
| down_state_dependence_slope    | -0.0153987 | 0.0311547 | -0.494265 | 0.621119   | cluster_OLS_subject_FE | 0.621119  |
| up_state_dependence_slope      | -0.214977  | 0.0804057 | -2.67366  | 0.00750287 | cluster_OLS_subject_FE | 0.0150057 |
| up_minus_down_state_dependence | -0.199579  | 0.0675958 | -2.95253  | 0.00315179 | cluster_OLS_subject_FE | 0.0126072 |
| up_minus_down_mean_success     |  0.0714108 | 0.0821299 |  0.869486 | 0.384581   | cluster_OLS_subject_FE | 0.512775  |

## Interpretation
- `expected_pain_c` tests whether success depends on the latent passive pain state for the down/reference condition.
- `is_up_c:expected_pain_c` tests whether the state-dependence slope differs between up- and down-regulation.
- `is_up_c` tests mean up-vs-down success after controlling for latent passive state, time/order, and subject effects.
- Brain-behavior links are exploratory; any inferential use requires explicit multiplicity correction.
