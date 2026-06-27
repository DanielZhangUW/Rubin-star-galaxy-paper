# Eq.54 + pS_color Integration Summary

Generated: 2026-06-26T20:12:23

## Inputs
- Eq.54 morphology/model posterior input: `outputs/dp2_cosmos_ps_v9_eq54prior.parquet`
- Color-only input: `outputs/dp2_cosmos_ps_color_v1.parquet`
- Join key: `object_id` (unique in both inputs).

## Formula
- `logLR_color = logit(pS_color)` with clipping epsilon `1e-6`.
- `logLR_total = logLR_morphology + logLR_color`.
- `pS_eq54prior_color = sigmoid(logLR_total + log_prior_odds_eq54)`.
- The Eq.54 magnitude prior is applied once, not once per band.

## Counts
- Eq.54 rows: 865,287
- pS_color rows: 588,841
- joined rows: 588,841
- valid color-feature rows: 524,494
- truth-labeled rows: 294,093

## Caveats
- This is a first-pass post-hoc likelihood-factor integration.
- It assumes the exploratory Random Forest `pS_color` is calibrated enough for logit conversion.
- The color classifier was not retrained in this run.
- Existing Eq.54 and pS_color parquet inputs were not modified.

Detailed scalar summary: `paper_convergence/results/section2_bayesian_method/eq54_color_integration_summary.csv`
Performance summary: `paper_convergence/results/section2_bayesian_method/eq54_color_integration_performance_by_rmag.csv`
