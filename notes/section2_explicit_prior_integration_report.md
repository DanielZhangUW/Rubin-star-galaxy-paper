# Section 2 Explicit-Prior Integration Report

Generated: 2026-06-24T19:16:51

## Definition used in this pass

- Stored v9 `pS_*` values are treated as pre-prior/model-based scores for this controlled task.
- The explicit prior is from `paper_convergence/tables/v9_ratio_data.csv`.
- The prior column is `log10_NG_NS = log10(N_G/N_S)` for `field=COSMOS`, `band=r`.
- Fitted mixture weights in `paper_convergence/tables/v9_fit_parameters.csv` are not treated as the explicit prior in this task.
- Single-band calibrated pS uses `logit(pS_b_pre) + log_prior_odds_eq54`.
- Multiband calibrated pS sums model logits first and applies `log_prior_odds_eq54` once.
- Old finite-mean `pS_gri` / `pS_ugrizy` figures remain draft diagnostics, not final Bayesian multiband posterior results.

## Summary
- input rows: 865,287
- rows assigned finite prior: 588,346
- rows outside/missing prior: 276,941
- derived output table: `outputs/dp2_cosmos_ps_v9_eq54prior.parquet`

## Generated outputs

### calibration_summary
- `paper_convergence/results/section2_bayesian_method/eq54_prior_calibration_summary.csv`
- `paper_convergence/results/section2_bayesian_method/eq54_prior_calibration_summary.md`

### fig2_4
- `paper_convergence/figures/section2_bayesian_method/fig2_4_cosmos_r_pS_map_16_26_eq54prior.png`
- `paper_convergence/figures/section2_bayesian_method/fig2_4_cosmos_r_pS_map_16_26_eq54prior.pdf`
- `paper_convergence/results/section2_bayesian_method/fig2_4_cosmos_r_pS_map_16_26_eq54prior_summary.csv`

### fig2_5
- `paper_convergence/figures/section2_bayesian_method/fig2_5_cosmos_r_pS_vs_extendedness_performance_eq54prior.png`
- `paper_convergence/figures/section2_bayesian_method/fig2_5_cosmos_r_pS_vs_extendedness_performance_eq54prior.pdf`
- `paper_convergence/results/section2_bayesian_method/fig2_5_cosmos_r_pS_vs_extendedness_performance_eq54prior_summary.csv`

### fig2_6
- `paper_convergence/figures/section2_bayesian_method/fig2_6_cosmos_multiband_r_gri_ugrizy_performance_eq54prior.png`
- `paper_convergence/figures/section2_bayesian_method/fig2_6_cosmos_multiband_r_gri_ugrizy_performance_eq54prior.pdf`
- `paper_convergence/results/section2_bayesian_method/fig2_6_cosmos_multiband_r_gri_ugrizy_performance_eq54prior_summary.csv`

### fig2_7
- `paper_convergence/figures/section2_bayesian_method/fig2_7_cosmos_ugrizy_method_color_color_2x4_eq54prior.png`
- `paper_convergence/figures/section2_bayesian_method/fig2_7_cosmos_ugrizy_method_color_color_2x4_eq54prior.pdf`
- `paper_convergence/results/section2_bayesian_method/fig2_7_cosmos_ugrizy_method_color_color_2x4_eq54prior_summary.csv`

