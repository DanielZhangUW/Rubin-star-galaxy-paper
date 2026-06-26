# Section 5 pS_color v1 Diagnostic Report

`pS_color` is a color-only Random Forest star probability.

Inputs and provenance:
- Classifier pickle: `paper_convergence/notebooks/SG-COSMOS-HST-ColorRFclassifier_all.pkl`
- DP2 analysis table: `outputs/dp2_cosmos_analysis_table.parquet`
- COSMOS2020 matched labels for evaluation only: `outputs/dp2_cosmos_cosmos2020_farmer_matched.parquet`
- Features: `ug, gr, ri, iz, zy`
- The classifier uses only dust-corrected colors and no morphology features.
- Label convention: star = 1, galaxy = 0; the score is `predict_proba(X)[:, 1]`.

Outputs:
- Derived pS_color table: `outputs/dp2_cosmos_ps_color_v1.parquet`
- Truth-label histogram: `paper_convergence/figures/section5_discussion/fig5_2_cosmos_ps_color_truth_hist_by_rmag.png`
- Performance diagnostic: `paper_convergence/figures/section5_discussion/fig5_3_cosmos_ps_color_performance_by_rmag.png`
- Method color-color diagnostic: `paper_convergence/figures/section5_discussion/fig5_4_cosmos_ps_color_method_color_color_2x4.png`
- Summary CSV: `paper_convergence/results/section5_discussion/ps_color_v1_summary.csv`

Performance summary:
- Total paper sample: 588,841
- Finite color-feature rows: 524,494
- Matched truth-label rows: 294,093
- Overall AUC: 0.9623
- Overall completeness at pS_color > 0.5: 0.6058
- Overall contamination at pS_color > 0.5: 0.0631
- Overall purity at pS_color > 0.5: 0.9369

Caveats:
- This is an exploratory notebook-level classifier made reproducible as a first-pass derived output.
- This is a color-only score, not morphology pS.
- This is not an Eq.54 morphology posterior.
- COSMOS2020 truth labels are used only for evaluation in this output.
- The pickle was saved with a different scikit-learn version than the current runtime; treat this as a reproducibility caveat until rerun in the original environment or retrained.
- Treat these figures as Section 5 / discussion diagnostics unless further validated.
