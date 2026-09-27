# Transition-level selected-pathway coupling sensitivity

This analysis directly models all **297 up-regulation transitions** rather than using each participant's nine-transition coupling coefficient as the inferential unit. It uses the archived rating-adjusted **AR(1)-prewhitened** trial estimates, because those are the transition-level rating-adjusted estimates included in the reproducibility package.

The analysis is conditional on the already selected dlPFC/IFJ-to-valuation pathway. It does **not** repeat the 116-correlation screening family and is not independent validation.

## Main result

- Equal-weight mean interaction across vmPFC, mOFC, and combined vmPFC/mOFC (unadjusted): **beta = -0.121**, participant-label permutation **p = 0.0136**.
- Covariate-adjusted mean interaction (source-slope moderation by regulation order, passive mean rating, and passive sensory slope): **beta = -0.163**, participant-label permutation **p = 0.0003**.
- Conditional Freedman-Lane permutation for the adjusted mean interaction: **p = 0.0102**.
- Participant-cluster bootstrap 95% CI for the adjusted mean interaction: **[-0.290, -0.054]**.
- Leave-one-participant-out adjusted mean interactions ranged from **-0.223 to -0.134**; all 33 remained negative.

## Target-specific pattern

The direct transition-level interaction was strongest and most stable for vmPFC and the combined vmPFC/mOFC definition. mOFC alone was weaker in the unadjusted model but became more negative after covariate adjustment. The equal-weight three-target mean is retained to mirror the reported composite construction; the target masks are overlapping/nested and are not independent replications.

## Mixed-model diagnostic

Random-slope mixed models produced interaction estimates in the same direction (vmPFC beta = -0.131; mOFC beta = -0.078; combined beta = -0.155). Only the vmPFC fit formally converged; the mOFC and combined fits produced convergence/boundary warnings. They are therefore retained as diagnostics rather than the primary inferential result.

## Interpretation

The selected success-coupling association survives a direct transition-level moderation analysis, so it is **not solely an artifact of correlating highly uncertain nine-transition participant coefficients**. This materially strengthens the reliability argument. It does not remove same-sample pathway selection, overlapping target definitions, hemodynamic limitations on lag interpretation, or the need for independent replication.
