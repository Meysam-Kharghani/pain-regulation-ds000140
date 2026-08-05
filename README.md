# Pain regulation in ds000140

[![Repository validation](https://github.com/Meysam-Kharghani/pain-regulation-ds000140/actions/workflows/validate.yml/badge.svg)](https://github.com/Meysam-Kharghani/pain-regulation-ds000140/actions/workflows/validate.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21752985.svg)](https://doi.org/10.5281/zenodo.21752985)

Analysis code, derived tables, and reporting assets for a secondary analysis of the public OpenNeuro ds000140 heat-pain regulation dataset.

**Release:** v1.0.0 (2026-08-02)  
**Source dataset:** [OpenNeuro ds000140](https://openneuro.org/datasets/ds000140)  
**Original study:** Woo et al. (2015), *PLOS Biology*, [doi:10.1371/journal.pbio.1002036](https://doi.org/10.1371/journal.pbio.1002036)

The evidence hierarchy is explicit. The strongest condition-level imaging result is down-minus-passive attenuation in bilateral S2/opercular and insular regions, accompanied by greater left dlPFC/IFJ activity. NPS regulation contrasts depend on the estimation path. Up-regulation success is associated with an exploratory, post-selection lagged left dlPFC/IFJ-to-vmPFC/mOFC composite after rating-period adjustment. Matched temporal controls, two conditional within-run null models, and an AR(1)-prewhitening sensitivity characterize that association. Run-separated DCM is a model-dependent sensitivity analysis and did not independently validate the selected coupling mechanism.

## Contents

```text
code/
  01_behavior/                 Behavioral reconstruction and models
  02_preprocessing_qc/         Preprocessing quality control
  03_glm/                      First- and group-level GLM analyses
  04_roi_signatures/           ROI and pain-signature analyses
  05_trialwise_connectivity/   Rating-adjusted FWL-LSS and coupling analyses
  06_dcm/                      Run-separated DCM/PEB analyses
  07_sensitivity_audits/       Robustness, null, and matched-control analyses

results/
  behavior/                    Behavioral results
  preprocessing_qc/            Run- and participant-level QC summaries
  glm/                         GLM inventories and group summaries
  roi_signatures/              ROI and signature results
  trialwise_connectivity/      Trialwise and pathway-level results
  sensitivity_audits/          Robustness, finite-cluster, and temporal-noise outputs
  reporting/                   Figures and exact machine-readable source data

docs/
  data_availability.md
  evidence_hierarchy.md
  methods_reproducibility.md
  reproducibility_workflows.md
  sensitivity_audit_manifest.md
  quality_assurance.md
  changelog.md
  file_manifest.tsv
  naming_conventions.md
  generated_artifact_naming.tsv
  content_review_report.tsv
```

## Quick validation

```bash
git clone https://github.com/Meysam-Kharghani/pain-regulation-ds000140.git
cd pain-regulation-ds000140
python -m venv .venv
source .venv/bin/activate
make validate
```

The validation target uses only the Python standard library and checks Python syntax, MATLAB primary-function naming, path conventions, JSON and TSV structure, repository-relative links, checksums, nonportable paths, transient caches, and internal working language.

Install `requirements_analysis.txt` only when running the Python analysis workflows; it is not required for repository validation.

## Reproduction levels

### Included outputs

Exact figure and table source data are stored in `results/reporting/`. These files are the authoritative values for the reported figures and tables. Earlier screening and diagnostic tables remain available for provenance but do not replace the rating-adjusted results.

### Dataset-level analyses

Download ds000140 and create the required derivatives. Set the dataset root explicitly:

```bash
export DS000140_ROOT=/absolute/path/to/ds000140
```

Scripts accept explicit input and output paths; use `python <script> --help` for stage-specific options. The primary trialwise ROI estimator uses rating-period nuisance regressors and Frisch–Waugh–Lovell residualization. The selected coupling analysis is implemented in:

```text
code/07_sensitivity_audits/run_matched_coupling_controls.py
```

### Full neuroimaging analyses

Full regeneration additionally requires fMRIPrep derivatives, the external atlas resources, the separately distributed NPS weights, MATLAB, and SPM25. The DCM workflow estimates each of the 288 retained runs independently for each of six five-node models, yielding 1,728 run-level DCMs. See `docs/methods_reproducibility.md`.

## Data boundaries

Raw MRI data, fMRIPrep derivatives, large image files, MATLAB model objects, third-party toolbox internals, restricted multivariate signature maps, and external atlas files are not redistributed. Their acquisition and expected locations are documented in `docs/data_availability.md`.

## Citation

Machine-readable citation metadata are available in [`CITATION.cff`](CITATION.cff). Cite this code collection and the original ds000140 dataset publication when reusing the materials.

## Authors

- Meysam Kharghani
- Elahe Gholami

Department of Cognitive Neuroscience, Faculty of Education and Psychology, University of Tabriz, Tabriz, Iran.

## License

Original code and documentation are released under the [MIT License](LICENSE). Source data and third-party resources retain their original terms.
