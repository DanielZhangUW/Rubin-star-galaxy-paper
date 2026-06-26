"""Metric helpers for paper-convergence validation figures."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_roc(y_true, score):
    """Compute ROC arrays and AUC with positive class encoded as 1."""

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    valid = np.isfinite(s) & np.isin(y, [0, 1])
    y = y[valid]
    s = s[valid]
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError("one class missing")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    change_idx = np.r_[np.where(s[1:] != s[:-1])[0], len(s) - 1]
    tps = np.cumsum(y == 1)[change_idx]
    fps = np.cumsum(y == 0)[change_idx]
    tpr = np.r_[0.0, tps / pos]
    fpr = np.r_[0.0, fps / neg]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def binary_operating_metrics(y_true, pred_star):
    """Star-positive metrics for a binary classifier."""

    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    pred = pd.Series(pred_star).astype(bool)
    valid = y.isin([0, 1]) & pred.notna()
    y = y[valid].astype(int)
    pred = pred[valid]
    truth_star = y.eq(1)
    truth_gal = y.eq(0)
    tp = int((truth_star & pred).sum())
    fn = int((truth_star & ~pred).sum())
    fp = int((truth_gal & pred).sum())
    tn = int((truth_gal & ~pred).sum())
    tpr = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    purity = tp / (tp + fp) if tp + fp else np.nan
    contamination = fp / (tp + fp) if tp + fp else np.nan
    return {
        "N_valid": int(valid.sum()),
        "N_star": int(truth_star.sum()),
        "N_galaxy": int(truth_gal.sum()),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "star_completeness": tpr,
        "galaxy_false_positive_rate": fpr,
        "star_purity": purity,
        "star_contamination": contamination,
        "step_auc": 0.5 * (1 + tpr - fpr) if np.isfinite(tpr) and np.isfinite(fpr) else np.nan,
    }


def density_score_train_apply(train, apply, bands, truth_col="truth_binary", pS_prefix="pS_", bins=50, epsilon=1e-6):
    """Empirical log-likelihood ratio score=sum log p(pS|star)-log p(pS|galaxy)."""

    edges = np.linspace(0, 1, bins + 1)
    score = pd.Series(0.0, index=apply.index)
    n_used = pd.Series(0, index=apply.index, dtype=int)
    diagnostics = []
    for band in bands:
        col = f"{pS_prefix}{band}"
        star_vals = pd.to_numeric(train.loc[train[truth_col].eq(1), col], errors="coerce").dropna().to_numpy(float)
        gal_vals = pd.to_numeric(train.loc[train[truth_col].eq(0), col], errors="coerce").dropna().to_numpy(float)
        ok = len(star_vals) > 0 and len(gal_vals) > 0
        diagnostics.append(
            {
                "band": band,
                "N_star_train": int(len(star_vals)),
                "N_galaxy_train": int(len(gal_vals)),
                "density_method": "histogram_density",
                "pS_bin_count": bins,
                "epsilon_floor": epsilon,
                "density_built_successfully": ok,
            }
        )
        if not ok:
            continue
        star_hist, _ = np.histogram(np.clip(star_vals, 0, 1), bins=edges, density=True)
        gal_hist, _ = np.histogram(np.clip(gal_vals, 0, 1), bins=edges, density=True)
        star_hist = np.maximum(star_hist.astype(float), epsilon)
        gal_hist = np.maximum(gal_hist.astype(float), epsilon)
        vals = pd.to_numeric(apply[col], errors="coerce")
        valid = np.isfinite(vals)
        idx = np.searchsorted(edges, np.clip(vals[valid].to_numpy(float), 0, 1), side="right") - 1
        idx = np.clip(idx, 0, len(star_hist) - 1)
        score.loc[valid.index[valid]] += np.log(star_hist[idx]) - np.log(gal_hist[idx])
        n_used.loc[valid.index[valid]] += 1
    score[n_used.eq(0)] = np.nan
    return score, n_used, diagnostics

