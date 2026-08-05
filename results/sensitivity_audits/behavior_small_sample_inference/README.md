# Finite-cluster behavioral inference

This directory reports finite-cluster sensitivity analyses for ten behavioral
models based on 33 participant clusters.

- `behavior_small_sample_inference.tsv` contains model estimates, CR2 standard
  errors, Satterthwaite degrees of freedom, and null-imposed Rademacher
  wild-cluster-bootstrap p values based on 9,999 replications.
- `r_session_info.txt` records the analysis inputs, random seed, R release, and
  package releases.

Intercept-only mean tests use the algebraically equivalent participant-cluster
sign-flip implementation. Bootstrap confidence intervals were not inverted;
the corresponding columns are intentionally empty.
