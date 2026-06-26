# Section 5 TRILEGAL Route A Prior Diagnostic

Route A uses Astro Data Lab / `lsst_sim.simdr2`, not a new TRILEGAL web run.

- Source table: `lsst_sim.simdr2`
- r magnitude column: `rmag`
- The table is treated as a precomputed LSST/TRILEGAL-like star simulation.
- `gc` is Galactic component and `label` is evolutionary phase, not galaxy/star class.
- TRILEGAL predicts stars only; galaxy counts in the implied prior are inferred from DP2 total counts.

Fields:
- COSMOS: 39,934 stars, area=2.2112 deg^2, catalog=`outputs/trilegal_cosmos_stars_v1.parquet`
- ECDFS: 31,745 stars, area=2.1887 deg^2, catalog=`outputs/trilegal_ecdfs_stars_v1.parquet`

Cleaned COSMOS prior diagnostic:
- Figure: `paper_convergence/figures/section5_discussion/fig5_7_cosmos_trilegal_prior_comparison_v1_cleaned.png`
- The cleaned figure omits TRILEGAL-implied prior points where the implied galaxy count is not physically valid.
- Masked COSMOS bin: `16.0-16.5`, where scaled TRILEGAL stars exceed DP2 total counts and the implied galaxy count was clipped to a small positive value.

Caveats:
- Area normalization uses rectangular RA/Dec footprints.
- Filter/magnitude systems may not perfectly match DP2 CModel r.
- DP2 completeness and selection cuts affect the implied galaxy counts.
- In bins where scaled TRILEGAL stars exceed DP2 total counts, the implied galaxy count is clipped to a small positive value; those implied-prior points are diagnostic only.
- COSMOS2020/HST matched labels are not complete at all magnitudes.
