# Trialwise connectivity results

This directory preserves the pathway-screening stage and its compact inputs. The screening stage preceded rating-period adjustment and is retained because the selected pathway's screening-family FDR value is part of the reported provenance.

The reported coupling estimate is not taken directly from these screening tables. It was recomputed from rating-adjusted FWL-LSS trial estimates and is stored in:

- `../sensitivity_audits/matched_coupling_controls/`
- `../reporting/figure_data/figure_04_source_data.tsv`
- `../reporting/table_data/main_table_04_source_data.tsv`

Use `code/05_trialwise_connectivity/run_trialwise_roi_connectivity.py` with `--include-rating-nuisance`, followed by `code/07_sensitivity_audits/run_matched_coupling_controls.py`, to regenerate the reported path from imaging derivatives.
