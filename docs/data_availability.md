# Data availability

## Public source data

The source dataset is OpenNeuro ds000140. The repository does not redistribute the raw MRI files. Obtain the dataset from OpenNeuro and set `DS000140_ROOT` to its absolute location.

## Included derived materials

The repository includes:

- cleaned behavioral and model tables;
- run and participant QC summaries;
- GLM inventories and group summaries;
- ROI and signature summaries;
- trialwise connectivity and pathway screening outputs;
- robustness and null-model outputs;
- exact figure and table source data;
- publication-quality PNG and vector PDF figures.

The authoritative values used in reported figures and tables are under `results/reporting/`.

## External resources

Full imaging regeneration requires resources that are not redistributed:

- fMRIPrep derivatives generated from ds000140;
- external atlas images used to construct the ROI masks;
- the Neurologic Pain Signature weight image;
- MATLAB and SPM25 for DCM/PEB.

The NPS weights are separately distributed under their own terms. The repository records the expected input role but does not copy the restricted image.

## Large intermediate data

Voxelwise images, resampled mask caches, trialwise intermediate images, and MATLAB DCM objects are omitted because they are large and regenerable. Compact inventories, diagnostics, and numerical summaries are included.

## Portability

No personal filesystem location is embedded in the public files. Paths in generated inventories should be supplied by command-line options or through `DS000140_ROOT`.
