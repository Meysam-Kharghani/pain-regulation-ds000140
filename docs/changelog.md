# Change log

## 2026-08-04

- Added CR2/Satterthwaite and wild-cluster-bootstrap sensitivity for behavioral models.
- Added AR(1)-prewhitened trialwise ROI estimates and repeated matched coupling controls.
- Added portable machine-readable result tables and reporting-source tables S7a and S7b.

## 1.0.0 — 2026-08-02

- Added exact reporting figures and machine-readable figure/table source data.
- Replaced the trialwise ROI estimator with the rating-adjusted FWL-LSS implementation.
- Added a matching trialwise NPS estimator.
- Added the nonzero within-run circular-shift null to the selected coupling controls.
- Consolidated the six run-separated DCM models under one configuration and driver.
- Removed obsolete focused and concatenated DCM paths and incompatible DCM summaries.
- Restored deterministic SHA-256-derived random seeds in signature analyses.
- Preserved the centered-percentile interpretation of nonzero state-dependence null distributions.
- Standardized public filenames, paths, and documentation.
- Normalized reporting PNG resolution metadata to 300 dpi and added automated validation.
- Simplified continuous integration so repository validation does not install analysis dependencies.
