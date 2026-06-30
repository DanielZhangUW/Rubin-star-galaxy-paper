# Current Paper-Convergence Review Manifest

This manifest lists the current Section 2 and Section 5 paper-convergence outputs ready for review. It intentionally references figures and small summaries only; source catalogs, derived parquet files, FITS files, and classifier pickle files are excluded from the upload set.

## Section 2 Eq.54 Prior Figures

### fig2_1_sg_prior_vs_rmag
- PNG: `figures/section2_bayesian_method/fig2_1_sg_prior_vs_rmag.png`
- Purpose: Compares the by-eye prior, COSMOS2020 matched-label ratio, extendedness ratio, and v9 empirical mixture-model prior as `log10(NG/NS)` versus r-band CModel magnitude.
- Caveat: Coverage proxy quantities are tracked separately from the `log10(NG/NS)` axis.

### fig2_2_cosmos_r_slice_fits_v9fit
- PNG: `figures/section2_bayesian_method/fig2_2_cosmos_r_slice_fits_v9fit.png`
- Purpose: Shows v9 r-band slice-fit diagnostics for the morphology likelihood model.
- Caveat: This is a fit diagnostic, not a final classification-performance plot.

### fig2_3_cosmos_ury_model2d_residuals_v9fit
- PNG: `figures/section2_bayesian_method/fig2_3_cosmos_ury_model2d_residuals_v9fit.png`
- Purpose: Shows u/r/y v9 model-residual diagnostics.
- Caveat: Residual structure should be inspected visually before choosing final manuscript panels.

### fig2_4_cosmos_r_pS_map_16_26_eq54prior
- PNG: `figures/section2_bayesian_method/fig2_4_cosmos_r_pS_map_16_26_eq54prior.png`
- Purpose: Shows the r-band Eq.54-prior-calibrated star-probability map for the COSMOS paper sample.
- Caveat: Uses derived Eq.54-prior columns; original stored pS parquet files are not modified.

### fig2_5_cosmos_r_pS_vs_extendedness_performance_eq54prior
- PNG: `figures/section2_bayesian_method/fig2_5_cosmos_r_pS_vs_extendedness_performance_eq54prior.png`
- Purpose: Compares r-band Eq.54-prior pS performance against r-band extendedness.
- Caveat: Threshold-level metrics should be read together with the summary CSV.

### fig2_6_cosmos_multiband_r_gri_ugrizy_performance_eq54prior
- PNG: `figures/section2_bayesian_method/fig2_6_cosmos_multiband_r_gri_ugrizy_performance_eq54prior.png`
- Purpose: Compares Eq.54-prior performance for r-only, gri, and ugrizy summed-logLR scores.
- Caveat: The explicit prior is applied once after summing model log-likelihood ratios.

### fig2_7_cosmos_ugrizy_method_color_color_2x4_eq54prior
- PNG: `figures/section2_bayesian_method/fig2_7_cosmos_ugrizy_method_color_color_2x4_eq54prior.png`
- Purpose: Shows color-color loci after classifying with the Eq.54-prior ugrizy method.
- Caveat: This is a method-label visualization, not an external-truth plot.

## Section 2 Eq.54 + pS_color Integration

### fig2_4_cosmos_r_pS_map_16_26_eq54prior_color
- PNG: `figures/section2_bayesian_method/fig2_4_cosmos_r_pS_map_16_26_eq54prior_color.png`
- Purpose: Shows the r-band posterior map after adding the color-only likelihood factor to the morphology log-likelihood ratio and applying the Eq.54 prior once.
- Caveat: This is a first-pass post-hoc integration using `logit(pS_color)` as a color likelihood ratio.

### fig2_5_cosmos_r_pS_vs_extendedness_performance_eq54prior_color
- PNG: `figures/section2_bayesian_method/fig2_5_cosmos_r_pS_vs_extendedness_performance_eq54prior_color.png`
- Purpose: Compares r-band extendedness, morphology-only Eq.54 r posterior, and Eq.54 r plus color performance.
- Caveat: The Eq.54 prior remains very conservative at faint magnitudes for threshold-based star selection.

### fig2_6_cosmos_multiband_r_gri_ugrizy_performance_eq54prior_color
- PNG: `figures/section2_bayesian_method/fig2_6_cosmos_multiband_r_gri_ugrizy_performance_eq54prior_color.png`
- Purpose: Compares color-integrated Eq.54 performance for r, gri, and ugrizy summed-logLR scores.
- Caveat: The color likelihood factor is added once to each morphology-band combination before applying the magnitude prior once.

### fig2_7_cosmos_ugrizy_method_color_color_2x4_eq54prior_color
- PNG: `figures/section2_bayesian_method/fig2_7_cosmos_ugrizy_method_color_color_2x4_eq54prior_color.png`
- Purpose: Shows color-color loci after classifying with the color-integrated Eq.54 ugrizy posterior.
- Caveat: Very few faint objects pass the `pS >= 0.5` star threshold because the Eq.54 prior strongly favors galaxies at the faint end.

### fig2_8_cosmos_eq54_color_roc_3bins
- PNG: `figures/section2_bayesian_method/fig2_8_cosmos_eq54_color_roc_3bins.png`
- Purpose: Shows star-positive ROC/AUC curves for the color-integrated Eq.54 r, gri, and ugrizy scores in three r magnitude bins.
- Caveat: ROC/AUC assesses ranking performance; fixed-threshold behavior at `pS > 0.5` is reported separately.

### fig2_9_cosmos_pS_vs_extendedness_roc_3bins_eq54prior_color
- PNG: `figures/section2_bayesian_method/fig2_9_cosmos_pS_vs_extendedness_roc_3bins_eq54prior_color.png`
- Purpose: Compares the r-band color-integrated Eq.54 ROC curve, morphology-only Eq.54 r ROC curve, and r-band extendedness binary operating-point line.
- Caveat: Extendedness is binary, so its display is an operating-point step line rather than a smooth ROC curve.

## Section 5.1 Red-Source Diagnostic

### fig5_1_cosmos_red_sources_r_pS_hist_vs_all_eq54prior
- PNG: `figures/section5_discussion/fig5_1_cosmos_red_sources_r_pS_hist_vs_all_eq54prior.png`
- Purpose: Compares Eq.54-prior r-band pS distributions for all COSMOS sources and red sources with `r-i > 1.4`.
- Caveat: The Eq.54 magnitude prior suppresses faint-end star probabilities and should be considered when interpreting red-source pS distributions.

### fig5_10_cosmos_eq54_vs_eq54_color_pS_hist_by_rmag
- PNG: `figures/section5_discussion/fig5_10_cosmos_eq54_vs_eq54_color_pS_hist_by_rmag.png`
- Purpose: Compares morphology-only Eq.54 pS, color-only `pS_color`, and color-integrated Eq.54 pS distributions by r magnitude bin.
- Caveat: Diagnostic comparison only; the color-integrated Section 2 figures are the main current outputs from this update.

## pS_color COSMOS And ECDFS Diagnostics

### fig5_2_cosmos_ps_color_truth_hist_by_rmag
- PNG: `figures/section5_discussion/fig5_2_cosmos_ps_color_truth_hist_by_rmag.png`
- Purpose: Shows COSMOS2020 truth-label histograms of the color-only `pS_color` score by r magnitude bin.
- Caveat: `pS_color` is color-only and is not the Eq.54 morphology posterior.

### fig5_3_cosmos_ps_color_performance_by_rmag
- PNG: `figures/section5_discussion/fig5_3_cosmos_ps_color_performance_by_rmag.png`
- Purpose: Summarizes COSMOS color-only classifier performance by r magnitude bin.
- Caveat: This uses the existing exploratory color Random Forest classifier.

### fig5_4_cosmos_ps_color_method_color_color_2x4
- PNG: `figures/section5_discussion/fig5_4_cosmos_ps_color_method_color_color_2x4.png`
- Purpose: Visualizes COSMOS color-only method labels in standard color-color planes.
- Caveat: Class labels come from `pS_color >= 0.5`, not morphology pS.

### fig5_2_ecdfs_ps_color_truth_hist_by_rmag
- PNG: `figures/section5_discussion/fig5_2_ecdfs_ps_color_truth_hist_by_rmag.png`
- Purpose: Shows ECDFS/HST truth-label histograms of the color-only `pS_color` score by r magnitude bin.
- Caveat: ECDFS/HST has smaller validation statistics than COSMOS/COSMOS2020.

### fig5_3_ecdfs_ps_color_performance_by_rmag
- PNG: `figures/section5_discussion/fig5_3_ecdfs_ps_color_performance_by_rmag.png`
- Purpose: Summarizes ECDFS color-only classifier performance by r magnitude bin.
- Caveat: Low star counts in some bins should be checked in the summary CSV.

### fig5_4_ecdfs_ps_color_method_color_color_2x4
- PNG: `figures/section5_discussion/fig5_4_ecdfs_ps_color_method_color_color_2x4.png`
- Purpose: Visualizes ECDFS color-only method labels in standard color-color planes.
- Caveat: Diagnostic comparison only; COSMOS remains the primary paper sample.

## TRILEGAL / lsst_sim.simdr2 Prior Diagnostic

### fig5_6_cosmos_trilegal_star_counts_comparison_v1
- PNG: `figures/section5_discussion/fig5_6_cosmos_trilegal_star_counts_comparison_v1.png`
- Purpose: Compares COSMOS TRILEGAL/lsst_sim simulated star counts with local count diagnostics.
- Caveat: Uses Astro Data Lab `lsst_sim.simdr2`, not a new TRILEGAL web run.

### fig5_7_cosmos_trilegal_prior_comparison_v1_cleaned
- PNG: `figures/section5_discussion/fig5_7_cosmos_trilegal_prior_comparison_v1_cleaned.png`
- Purpose: Compares COSMOS v9 empirical, matched-label, and physically valid TRILEGAL-implied `log10(NG/NS)` prior curves.
- Caveat: The `16.0-16.5` bin is masked because scaled TRILEGAL stars exceed DP2 total counts there.

### fig5_8_ecdfs_trilegal_star_counts_comparison_v1
- PNG: `figures/section5_discussion/fig5_8_ecdfs_trilegal_star_counts_comparison_v1.png`
- Purpose: Provides an ECDFS comparison for the TRILEGAL/lsst_sim star-count diagnostic.
- Caveat: ECDFS is secondary and uses its full DP2 analysis-table footprint for this field-level comparison.

### fig5_9_cosmos_ecdfs_trilegal_star_counts_comparison_v1
- PNG: `figures/section5_discussion/fig5_9_cosmos_ecdfs_trilegal_star_counts_comparison_v1.png`
- Purpose: Compares COSMOS and ECDFS TRILEGAL/lsst_sim star-count trends.
- Caveat: This is a diagnostic check on field-to-field star-count expectations, not a replacement for the empirical v9 prior.
