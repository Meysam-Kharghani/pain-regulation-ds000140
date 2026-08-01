# Naming conventions

This repository uses deterministic, role-based names.

## General rules

- Repository paths use lowercase `snake_case`.
- Standard community files are the only uppercase-name exceptions: `README.md`, `AUTHORS.md`, `CITATION.cff`, `CONTRIBUTING.md`, `LICENSE`, and `Makefile`.
- Scientific abbreviations use lowercase in paths: `dcm`, `peb`, `glm`, `roi`, `nps`, `lss`, `fc`, `ec`, `fdr`, and `qc`.
- Figure and table numbers are zero-padded in filenames, for example `figure_01.pdf` and `main_table_01_source_data.tsv`. Display labels remain “Figure 1” and “Table 1”.
- Filenames do not contain project-status labels such as `final`, `revision`, or version suffixes. Version history is managed by Git.

## Code files

Executable scripts use an action-first pattern: `<verb>_<object>[_<qualifier>].<ext>`. Common verbs are `prepare`, `run`, `estimate`, `build`, `validate`, `audit`, `compare`, `extract`, `diagnose`, and `verify`.

MATLAB files that define a primary function have the same filename and function name. Canonical mathematical and SPM object identifiers such as `A`, `B`, `C`, `DCM`, `PEB`, `BMA`, and `BMR` may retain uppercase notation inside code; path names remain lowercase. Configuration providers begin with `get_`; validation functions begin with `validate_`; output audits begin with `audit_`; workflow entry points begin with `run_`.

## Result files

Generated results use a content-first noun phrase followed by an artifact suffix such as `_results`, `_summary`, `_tests`, `_inventory`, `_report`, `_manifest`, or `_source_data`. Formula syntax and implementation shorthand are not used in filenames.

## Generated directories

Generated directories follow the same lowercase convention. Examples include `estimated_dcms`, `dcm_structs`, `peb_subjects`, and `peb_group`.
