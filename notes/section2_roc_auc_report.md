# Section 2 ROC/AUC Report

This report accompanies the Fig 2.8 and Fig 2.9 ROC/AUC diagnostics for the current Eq.54 + pS_color outputs.

## Purpose

Existing Fig 2.5 and Fig 2.6 summarize fixed-threshold performance at `pS > 0.5`. ROC/AUC is threshold-independent and is useful because the Eq.54 magnitude prior makes `pS > 0.5` very conservative at the faint end.

AUC should be used to assess ranking performance. Threshold metrics show one operating point.

## Inputs

- Derived pS table: `outputs/dp2_cosmos_ps_v9_eq54prior_color.parquet`
- Positive class: COSMOS2020 truth star (`truth_binary = 1`).
- Negative class: COSMOS2020 truth galaxy (`truth_binary = 0`).
- Magnitude column: `cmodel_mag_r`.

## Figures

- Fig 2.8: ROC curves for `pS_r_eq54prior_color`, `pS_gri_eq54prior_color`, and `pS_ugrizy_eq54prior_color`.
- Fig 2.9: ROC curves for `pS_r_eq54prior_color` and morphology-only `pS_r_eq54prior`, with the r-band extendedness operating point where available.

## Extendedness Operating Point

The r-band extendedness operating point was read from the existing derived performance summary `paper_convergence/results/section2_bayesian_method/eq54_color_integration_performance_by_rmag.csv`.

## Caveats

- These diagnostics do not retrain or recalibrate `pS_color`.
- Existing Eq.54 and color-integrated parquet inputs were not modified.
- Fig 2.8/2.9 use FPR on the x-axis for ROC consistency; fixed-threshold contamination/purity are recorded in the CSV summaries.

## Summary Rows

- Fig 2.8 summary rows: 9
- Fig 2.9 summary rows, including extendedness rows if present: 9
