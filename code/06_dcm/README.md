# Sessionwise DCM workflow

The DCM files use role-specific names.

## Entry points

- `run_sessionwise_dcm_variant.m`: run one model variant in smoke-test or full mode.
- `run_all_sessionwise_dcm_variants.m`: run all six model variants.
- `run_reduced_group_peb_sensitivity.m`: fit the prespecified reduced group-level PEB sensitivity models.

## Configuration

- `get_sessionwise_dcm_base_config.m`: shared paths and analysis constants.
- `get_sessionwise_dcm_variant_config.m`: model-family and ROI definitions for one variant.
- `get_sessionwise_dcm_config.m`: merged configuration used by the pipeline.

## Validation and execution

- `validate_sessionwise_dcm_inputs.m`: validate the selected variant, input tables, ROI definitions, and masks before estimation.
- `run_sessionwise_dcm_pipeline.m`: construct and estimate run-level DCMs and hierarchical PEB models.
- `audit_sessionwise_dcm_outputs.m`: validate completed output counts and posterior objects.

The primary MATLAB function in each file matches the filename exactly. Generated DCM paths use lowercase names, including `dcm_structs`, `estimated_dcms`, `peb_subjects`, and `peb_group`.
