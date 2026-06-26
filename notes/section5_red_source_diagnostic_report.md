# Section 5.1 Red-Source Diagnostic

This is the Section 5.1 red-source diagnostic for the COSMOS paper sample.

Definitions:
- Paper sample: `16 < cmodel_mag_r < 26`.
- Red sources: dust-corrected `r-i > 1.4`, using the existing `ri` column from the COSMOS analysis table.
- Eq.54-prior score: `pS_r_eq54prior` from `outputs/dp2_cosmos_ps_v9_eq54prior.parquet`.

Outputs:
- Figure PNG: `paper_convergence/figures/section5_discussion/fig5_1_cosmos_red_sources_r_pS_hist_vs_all_eq54prior.png`
- Figure PDF: `paper_convergence/figures/section5_discussion/fig5_1_cosmos_red_sources_r_pS_hist_vs_all_eq54prior.pdf`
- Summary CSV: `paper_convergence/results/section5_discussion/fig5_1_cosmos_red_sources_r_pS_hist_vs_all_eq54prior_summary.csv`
- Summary MD: `paper_convergence/results/section5_discussion/fig5_1_cosmos_red_sources_r_pS_hist_vs_all_eq54prior_summary.md`

Status notes:
- The old v8/stored-pS Section 5.1 figure remains diagnostic only and was not overwritten.
- Old PNG exists: `paper_convergence/figures/section5_discussion/fig5_1_cosmos_red_sources_r_pS_hist_vs_all.png` = True
- Old PDF exists: `paper_convergence/figures/section5_discussion/fig5_1_cosmos_red_sources_r_pS_hist_vs_all.pdf` = True
- Because the Eq.54 prior strongly suppresses faint-end star probabilities, red-source pS distributions should be interpreted with that prior effect in mind.

Sample counts:
- Paper-sample rows: 588,841
- Red-source rows: 4,175
- Rows with finite Eq.54 pS in paper sample: 588,096
- Red-source rows with finite Eq.54 pS: 4,167
