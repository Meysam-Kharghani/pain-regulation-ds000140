# Quality assurance

The public tree is checked for:

- Python syntax;
- valid JSON and rectangular TSV files;
- complete reporting-source mappings;
- repository-relative Markdown links;
- missing or empty files;
- compiled caches and temporary artifacts;
- personal filesystem locations;
- internal conversational or automated-assistant language;
- project-status labels in filenames;
- checksums for all stable files; mutable release metadata (`README.md`, `CITATION.cff`, and `docs/changelog.md`) are validated structurally but intentionally excluded from the checksum manifest.

Scientific cross-checks include:

- 3,201 behavioral trials and 33 participants;
- 288 retained imaging runs, including all 66 regulation runs;
- 660 regulation trials and 594 within-run transitions for the rating-adjusted coupling analysis;
- reported forward coupling correlation of −0.545718907902466;
- shared-source and circular-shift empirical probabilities of 0.0013997200559888023;
- six run-separated DCM models with 288 runs per model.

The exact values plotted or tabulated are stored under `results/reporting/`.
