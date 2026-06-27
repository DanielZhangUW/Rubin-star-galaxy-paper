# Section 2 Eq.54 + pS_color Integration Report

Generated: 2026-06-26T20:12:27

## Purpose

`pS_color` was previously a diagnostic color-only star probability. This run integrates it into the final likelihood/posterior calculation as an additional color likelihood-ratio factor.

## Implementation

- `pS_color` is converted to `logLR_color = logit(pS_color)` with clipping at `1e-6` and `1 - 1e-6`.
- Morphology/model log-likelihood ratios are taken from `outputs/dp2_cosmos_ps_v9_eq54prior.parquet`.
- Combined log-likelihood ratios use `logLR_total = logLR_morphology + logLR_color`.
- The Eq.54 magnitude prior is applied once: `pS = sigmoid(logLR_total + log_prior_odds_eq54)`.
- No stored Eq.54 or pS_color input parquet was modified.
- The color classifier was not retrained.

## Caveats

- This is a first-pass post-hoc likelihood-factor integration.
- It assumes the exploratory Random Forest `pS_color` is calibrated enough for logit conversion.
- A cleaner cross-validated color classifier may be needed before final paper claims.
- `pS_color` is color-only and does not use morphology.

## Outputs

- `outputs/dp2_cosmos_ps_v9_eq54prior_color.parquet`
- `paper_convergence/results/section2_bayesian_method/eq54_color_integration_summary.csv`
- `paper_convergence/results/section2_bayesian_method/eq54_color_integration_summary.md`
- `paper_convergence/results/section2_bayesian_method/eq54_color_integration_performance_by_rmag.csv`
- `paper_convergence/figures/section2_bayesian_method/fig2_4_cosmos_r_pS_map_16_26_eq54prior_color.png`
- `paper_convergence/figures/section2_bayesian_method/fig2_4_cosmos_r_pS_map_16_26_eq54prior_color.pdf`
- `paper_convergence/figures/section2_bayesian_method/fig2_5_cosmos_r_pS_vs_extendedness_performance_eq54prior_color.png`
- `paper_convergence/figures/section2_bayesian_method/fig2_5_cosmos_r_pS_vs_extendedness_performance_eq54prior_color.pdf`
- `paper_convergence/figures/section2_bayesian_method/fig2_6_cosmos_multiband_r_gri_ugrizy_performance_eq54prior_color.png`
- `paper_convergence/figures/section2_bayesian_method/fig2_6_cosmos_multiband_r_gri_ugrizy_performance_eq54prior_color.pdf`
- `paper_convergence/figures/section2_bayesian_method/fig2_7_cosmos_ugrizy_method_color_color_2x4_eq54prior_color.png`
- `paper_convergence/figures/section2_bayesian_method/fig2_7_cosmos_ugrizy_method_color_color_2x4_eq54prior_color.pdf`
- `paper_convergence/results/section2_bayesian_method/fig2_7_cosmos_ugrizy_method_color_color_2x4_eq54prior_color_summary.csv`
- `paper_convergence/figures/section5_discussion/fig5_10_cosmos_eq54_vs_eq54_color_pS_hist_by_rmag.png`
- `paper_convergence/figures/section5_discussion/fig5_10_cosmos_eq54_vs_eq54_color_pS_hist_by_rmag.pdf`
