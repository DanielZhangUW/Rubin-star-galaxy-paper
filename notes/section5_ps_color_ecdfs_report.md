# ECDFS pS_color v1 Validation Report

This is an ECDFS external-field validation of the same color-only Random Forest classifier used for COSMOS pS_color v1.

Inputs and provenance:
- Classifier pickle: `paper_convergence/notebooks/SG-COSMOS-HST-ColorRFclassifier_all.pkl`
- ECDFS DP2 analysis table: `outputs/dp2_ecdfs_analysis_table.parquet`
- HST matched labels for evaluation only: `outputs/dp2_ecdfs_hst_matched.parquet`
- Features: `ug, gr, ri, iz, zy`
- Feature order is fixed as `ug, gr, ri, iz, zy`.
- No retraining was done.
- The classifier uses only dust-corrected colors and no morphology features.
- Label convention: star = 1, galaxy = 0; score is `predict_proba(X)[:, 1]`.

Outputs:
- Derived pS_color table: `outputs/dp2_ecdfs_ps_color_v1.parquet`
- Truth-label histogram: `paper_convergence/figures/section5_discussion/fig5_2_ecdfs_ps_color_truth_hist_by_rmag.png`
- Performance diagnostic: `paper_convergence/figures/section5_discussion/fig5_3_ecdfs_ps_color_performance_by_rmag.png`
- Method color-color diagnostic: `paper_convergence/figures/section5_discussion/fig5_4_ecdfs_ps_color_method_color_color_2x4.png`
- Summary CSV: `paper_convergence/results/section5_discussion/ps_color_v1_ecdfs_summary.csv`
- COSMOS vs ECDFS comparison CSV: `paper_convergence/results/section5_discussion/ps_color_v1_cosmos_vs_ecdfs_summary.csv`

Performance summary:
- Total ECDFS paper sample: 538,626
- Finite color-feature rows: 370,224
- HST truth-labeled rows: 6,372
- Overall AUC: 0.9307
- Overall completeness at pS_color > 0.5: 0.6622
- Overall contamination at pS_color > 0.5: 0.2794
- Overall purity at pS_color > 0.5: 0.7206

Caveats:
- This is a color-only score, not morphology pS.
- This is not an Eq.54 morphology posterior.
- The classifier originates from an exploratory notebook/pickle.
- ECDFS/HST has a much smaller truth-label sample than COSMOS/COSMOS2020, so per-bin metrics are noisier.
- The pickle was saved with a different scikit-learn version than the current runtime; treat this as a reproducibility caveat until rerun in the original environment or retrained.
- Treat these figures as Section 5 / discussion diagnostics unless further validated.
