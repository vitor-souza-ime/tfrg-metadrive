# Traffic–friction experiment: recorded results

- Recorded completed episodes: 864
- MetaDrive step observations: 282,517
- Independent scenario seeds: 18
- Map families: CSC, SCS
- Traffic densities: 0.00, 0.18
- Vehicle collisions and lane departures are separate outcomes.
- Traffic density is a simulator setting, not an observed vehicle flow in vehicles/hour.
- Actual active vehicle counts and within-25-m exposure are measured each step.
- Traffic positions are used for logging, not given to the speed governors.
- Interval estimates resample whole scenario seeds, not time steps.
- Bootstrap 95% intervals are supplied in tables/paired_effects.csv.
- Trip times are conditional on successful completion (selection bias is possible).
- All reported data originate from local MetaDrive simulator execution, not physical experiments.
- Do not infer superior methods without checking the paired effects and their uncertainty.
