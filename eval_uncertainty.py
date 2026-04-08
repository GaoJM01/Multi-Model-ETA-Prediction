#!/usr/bin/env python
"""
Uncertainty Quantification via Ensemble Conformal Prediction.

Uses the 15-seed MSTGN-MLP2 ensemble to:
1. Compute ensemble mean (point estimate) and std (epistemic uncertainty)
2. Apply split conformal prediction for calibrated prediction intervals
3. Evaluate stratified calibration metrics by duration bin and sailing progress
4. Propose Duration-Stratified Calibration Error (DSCE) metric

Output: Tables and figures for paper.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# discord notification
import urllib.request

WEBHOOK_URL = "https://discord.com/api/webhooks/1477180434474336266/HQSS_BKlo1Ib-rzwItazg8ay0To2l-GR1GprnTvlPVpLxCxD9xu6_iEPsD9aBd4iHNgX"


def notify_discord(msg):
    try:
        data = json.dumps({"content": msg}).encode()
        req = urllib.request.Request(WEBHOOK_URL, data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def calc_metrics(y_pred, y_true):
    """Compute MAE, RMSE, MAPE, MedAE, R²."""
    residuals = y_pred - y_true
    abs_err = np.abs(residuals)
    mae = np.mean(abs_err)
    rmse = np.sqrt(np.mean(residuals ** 2))
    mask = y_true > 24
    mape = np.mean(np.abs(residuals[mask] / y_true[mask])) * 100 if mask.sum() > 0 else float('nan')
    medae = np.median(abs_err)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MedAE': medae, 'R2': r2}


def within_threshold(y_pred, y_true, thresholds_h=(6, 12, 24, 48)):
    """Fraction of predictions within ±threshold hours of truth."""
    abs_err = np.abs(y_pred - y_true)
    return {f'within_{t}h': float(np.mean(abs_err <= t)) for t in thresholds_h}


def conformal_calibrate(residuals_cal, alpha):
    """Compute conformal quantile from calibration residuals.

    residuals_cal: array of |y_true - y_pred| / sigma on calibration set
    alpha: desired miscoverage rate (e.g. 0.1 for 90% coverage)
    Returns: quantile q such that P(|y - ŷ| <= q * sigma) >= 1 - alpha
    """
    n = len(residuals_cal)
    # Finite-sample correction: use ceil((n+1)(1-alpha))/n quantile
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(residuals_cal, q_level))


def compute_interval_metrics(y_true, y_lower, y_upper):
    """Compute PICP and MPIW for prediction intervals."""
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    picp = np.mean(covered)
    mpiw = np.mean(y_upper - y_lower)
    return float(picp), float(mpiw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble_dir', default='output/ensemble')
    parser.add_argument('--output_dir', default='output/uncertainty')
    parser.add_argument('--top_k', type=int, default=7, help='Top-K models for ensemble')
    parser.add_argument('--coverages', type=float, nargs='+',
                        default=[0.90, 0.95], help='Target coverage levels')
    args = parser.parse_args()

    ensemble_dir = Path(args.ensemble_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 1. Load all ensemble predictions
    # =========================================================
    seed_dirs = sorted([d for d in ensemble_dir.iterdir()
                        if d.is_dir() and d.name.startswith('seed')])

    # Rank by val_loss to select top-K
    seed_info = []
    for sd in seed_dirs:
        rfile = sd / 'results.json'
        if rfile.exists():
            with open(rfile) as f:
                r = json.load(f)
            seed_info.append((sd, r.get('best_val_loss', float('inf'))))
    seed_info.sort(key=lambda x: x[1])

    # Load top-K predictions
    top_dirs = [sd for sd, _ in seed_info[:args.top_k]]
    y_true = None
    preds_topk = []  # (K, N)
    print(f"Loading top-{args.top_k} ensemble predictions...")
    for sd in top_dirs:
        d = np.load(sd / 'predictions.npz')
        if y_true is None:
            y_true = d['y_true']
        preds_topk.append(d['y_pred'])
        mae_i = np.mean(np.abs(d['y_pred'] - y_true))
        print(f"  {sd.name}: MAE={mae_i:.2f}h, val_loss={seed_info[[s for s,_ in seed_info].index(sd)][1]:.6f}")
    preds_topk = np.array(preds_topk)  # (K, N)

    # Also load ALL 15 predictions for comparison
    preds_all = []
    for sd, _ in seed_info:
        d = np.load(sd / 'predictions.npz')
        preds_all.append(d['y_pred'])
    preds_all = np.array(preds_all)  # (15, N)

    N = len(y_true)
    print(f"\nTest samples: {N:,}")

    # =========================================================
    # 2. Ensemble point estimate + uncertainty
    # =========================================================
    y_mean = np.mean(preds_topk, axis=0)   # point estimate
    y_std = np.std(preds_topk, axis=0)      # epistemic uncertainty (spread)

    # Also compute from all 15
    y_mean_all = np.mean(preds_all, axis=0)
    y_std_all = np.std(preds_all, axis=0)

    print(f"\nEnsemble-{args.top_k} uncertainty stats:")
    print(f"  Mean sigma: {y_std.mean():.2f}h, Median sigma: {np.median(y_std):.2f}h")
    print(f"  Max sigma: {y_std.max():.2f}h, Min sigma: {y_std.min():.2f}h")

    # =========================================================
    # 3. Extended point-prediction metrics
    # =========================================================
    y_pred = np.maximum(y_mean, 0)
    metrics_point = calc_metrics(y_pred, y_true)
    metrics_threshold = within_threshold(y_pred, y_true)
    metrics_point.update(metrics_threshold)

    print(f"\n{'='*60}")
    print(f"Extended Metrics (Ens-{args.top_k} mean):")
    print(f"{'='*60}")
    for k, v in metrics_point.items():
        if 'within' in k:
            print(f"  {k}: {v*100:.1f}%")
        else:
            print(f"  {k}: {v:.2f}" + ("%" if k == 'MAPE' else "h" if k in ['MAE','RMSE','MedAE'] else ""))

    # =========================================================
    # 4. Split Conformal Prediction
    # =========================================================
    # Use random 50% of test set for calibration, rest for evaluation
    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    n_cal = N // 2
    cal_idx, eval_idx = idx[:n_cal], idx[n_cal:]

    # Normalized residuals on calibration set
    sigma_cal = np.maximum(y_std[cal_idx], 1e-3)  # avoid division by zero
    abs_residuals_cal = np.abs(y_true[cal_idx] - y_pred[cal_idx]) / sigma_cal

    # Also compute raw (non-normalized) residuals for non-adaptive intervals
    abs_residuals_raw_cal = np.abs(y_true[cal_idx] - y_pred[cal_idx])

    print(f"\nConformal Prediction (cal={n_cal:,}, eval={N-n_cal:,}):")

    conformal_results = {}
    for coverage in args.coverages:
        alpha = 1 - coverage

        # Adaptive intervals (normalized by ensemble sigma)
        q_adaptive = conformal_calibrate(abs_residuals_cal, alpha)
        sigma_eval = np.maximum(y_std[eval_idx], 1e-3)
        y_lower_a = y_pred[eval_idx] - q_adaptive * sigma_eval
        y_upper_a = y_pred[eval_idx] + q_adaptive * sigma_eval
        picp_a, mpiw_a = compute_interval_metrics(y_true[eval_idx], y_lower_a, y_upper_a)

        # Non-adaptive intervals (constant width)
        q_fixed = conformal_calibrate(abs_residuals_raw_cal, alpha)
        y_lower_f = y_pred[eval_idx] - q_fixed
        y_upper_f = y_pred[eval_idx] + q_fixed
        picp_f, mpiw_f = compute_interval_metrics(y_true[eval_idx], y_lower_f, y_upper_f)

        print(f"\n  Target coverage: {coverage*100:.0f}%")
        print(f"  Adaptive (σ-scaled):  PICP={picp_a*100:.1f}%, MPIW={mpiw_a:.1f}h, q={q_adaptive:.2f}")
        print(f"  Fixed-width:          PICP={picp_f*100:.1f}%, MPIW={mpiw_f:.1f}h, q={q_fixed:.1f}h")

        conformal_results[f'{coverage*100:.0f}'] = {
            'adaptive': {'PICP': picp_a, 'MPIW': mpiw_a, 'q': q_adaptive},
            'fixed': {'PICP': picp_f, 'MPIW': mpiw_f, 'q': q_fixed},
        }

    # =========================================================
    # 5. Stratified Calibration Analysis
    # =========================================================
    print(f"\n{'='*60}")
    print("Stratified Calibration Analysis")
    print(f"{'='*60}")

    # Duration bins
    duration_bins = [(0, 100, '0-100'), (100, 200, '100-200'),
                     (200, 400, '200-400'), (400, 720, '400-720')]

    # Sailing progress bins (remaining time as fraction of typical voyage ~360h)
    # Instead, use absolute remaining time bins which are more interpretable
    progress_bins = [(0, 24, '<1 day'), (24, 72, '1-3 days'),
                     (72, 168, '3-7 days'), (168, 336, '7-14 days'),
                     (336, 720, '>14 days')]

    # Use full test set for stratified analysis (not split)
    target_coverage = 0.90
    alpha = 1 - target_coverage

    # Re-calibrate on full calibration half for 90% coverage
    q_adaptive_90 = conformal_calibrate(abs_residuals_cal, alpha)
    sigma_full = np.maximum(y_std, 1e-3)
    y_lower_full = y_pred - q_adaptive_90 * sigma_full
    y_upper_full = y_pred + q_adaptive_90 * sigma_full

    strata_results = {}

    # --- Duration-stratified ---
    print(f"\n--- Duration-Stratified (90% target, adaptive) ---")
    print(f"{'Bin':>12s} {'N':>8s} {'PICP':>7s} {'MPIW(h)':>8s} {'MAE(h)':>7s} {'MedAE':>7s} {'σ_mean':>7s}")
    for lo, hi, label in duration_bins:
        mask = (y_true >= lo) & (y_true < hi)
        n_bin = mask.sum()
        if n_bin < 10:
            continue
        picp_bin, mpiw_bin = compute_interval_metrics(y_true[mask], y_lower_full[mask], y_upper_full[mask])
        mae_bin = np.mean(np.abs(y_pred[mask] - y_true[mask]))
        medae_bin = np.median(np.abs(y_pred[mask] - y_true[mask]))
        sigma_bin = y_std[mask].mean()
        print(f"{label:>12s} {n_bin:>8,d} {picp_bin*100:>6.1f}% {mpiw_bin:>8.1f} {mae_bin:>7.1f} {medae_bin:>7.1f} {sigma_bin:>7.1f}")
        strata_results[f'dur_{label}'] = {
            'N': int(n_bin), 'PICP': picp_bin, 'MPIW': mpiw_bin,
            'MAE': float(mae_bin), 'MedAE': float(medae_bin), 'sigma_mean': float(sigma_bin)
        }

    # --- Progress-stratified ---
    print(f"\n--- Remaining-Time-Stratified (90% target, adaptive) ---")
    print(f"{'Bin':>12s} {'N':>8s} {'PICP':>7s} {'MPIW(h)':>8s} {'MAE(h)':>7s} {'RPIW%':>7s} {'σ_mean':>7s}")
    for lo, hi, label in progress_bins:
        mask = (y_true >= lo) & (y_true < hi)
        n_bin = mask.sum()
        if n_bin < 10:
            continue
        picp_bin, mpiw_bin = compute_interval_metrics(y_true[mask], y_lower_full[mask], y_upper_full[mask])
        mae_bin = np.mean(np.abs(y_pred[mask] - y_true[mask]))
        median_y = np.median(y_true[mask])
        rpiw = mpiw_bin / median_y * 100 if median_y > 0 else float('nan')  # relative interval width
        sigma_bin = y_std[mask].mean()
        print(f"{label:>12s} {n_bin:>8,d} {picp_bin*100:>6.1f}% {mpiw_bin:>8.1f} {mae_bin:>7.1f} {rpiw:>6.1f}% {sigma_bin:>7.1f}")
        strata_results[f'rem_{label}'] = {
            'N': int(n_bin), 'PICP': picp_bin, 'MPIW': mpiw_bin,
            'MAE': float(mae_bin), 'RPIW': float(rpiw), 'sigma_mean': float(sigma_bin)
        }

    # =========================================================
    # 6. Duration-Stratified Calibration Error (DSCE)
    # =========================================================
    # DSCE = max over strata of |PICP_stratum - target_coverage|
    # Lower is better — measures worst-case miscalibration across voyage durations
    picp_strata = [v['PICP'] for k, v in strata_results.items() if k.startswith('dur_')]
    dsce = max(abs(p - target_coverage) for p in picp_strata) if picp_strata else float('nan')

    # Also compute mean calibration error
    mce = np.mean([abs(p - target_coverage) for p in picp_strata]) if picp_strata else float('nan')

    print(f"\n{'='*60}")
    print(f"Duration-Stratified Calibration Error (DSCE)")
    print(f"{'='*60}")
    print(f"  Target coverage: {target_coverage*100:.0f}%")
    print(f"  DSCE (max |PICP - target|): {dsce*100:.2f}pp")
    print(f"  MCE  (mean |PICP - target|): {mce*100:.2f}pp")

    # Also for progress strata
    picp_progress = [v['PICP'] for k, v in strata_results.items() if k.startswith('rem_')]
    dsce_progress = max(abs(p - target_coverage) for p in picp_progress) if picp_progress else float('nan')
    mce_progress = np.mean([abs(p - target_coverage) for p in picp_progress]) if picp_progress else float('nan')
    print(f"\n  Progress-stratified:")
    print(f"  DSCE: {dsce_progress*100:.2f}pp")
    print(f"  MCE:  {mce_progress*100:.2f}pp")

    # =========================================================
    # 7. Operational metrics: buffer time recommendations
    # =========================================================
    print(f"\n{'='*60}")
    print("Operational Buffer Time Analysis")
    print(f"{'='*60}")

    # For each remaining-time bin, compute the buffer needed for 90%/95% reliability
    for lo, hi, label in progress_bins:
        mask = (y_true >= lo) & (y_true < hi)
        n_bin = mask.sum()
        if n_bin < 10:
            continue
        errors = y_true[mask] - y_pred[mask]  # positive = late arrival (underestimate)
        # Buffer for 90% on-time: 90th percentile of (y_true - y_pred)
        buf_90 = np.percentile(errors, 90)
        buf_95 = np.percentile(errors, 95)
        late_rate = np.mean(errors > 0) * 100
        print(f"  {label:>12s} (N={n_bin:,}): late_rate={late_rate:.1f}%, "
              f"buffer_90={buf_90:.1f}h, buffer_95={buf_95:.1f}h")

    # =========================================================
    # 8. Uncertainty-Error Correlation
    # =========================================================
    abs_err = np.abs(y_pred - y_true)
    # Spearman rank correlation between sigma and |error|
    from scipy.stats import spearmanr
    rho, pval = spearmanr(y_std, abs_err)
    print(f"\nUncertainty-Error Correlation:")
    print(f"  Spearman ρ(σ, |error|) = {rho:.4f} (p={pval:.2e})")

    # Binned analysis: sort by sigma quintiles
    print(f"\n  Uncertainty quintile analysis:")
    print(f"  {'Quintile':>10s} {'σ range':>16s} {'MAE(h)':>8s} {'PICP':>7s}")
    quintile_edges = np.percentile(y_std, [0, 20, 40, 60, 80, 100])
    for i in range(5):
        mask = (y_std >= quintile_edges[i]) & (y_std < quintile_edges[i+1] + 1e-6)
        mae_q = np.mean(abs_err[mask])
        picp_q, _ = compute_interval_metrics(y_true[mask], y_lower_full[mask], y_upper_full[mask])
        print(f"  Q{i+1:>9d} [{quintile_edges[i]:6.1f}-{quintile_edges[i+1]:6.1f}] {mae_q:>8.1f} {picp_q*100:>6.1f}%")

    # =========================================================
    # 9. Save all results
    # =========================================================
    all_results = {
        'point_metrics': {k: float(v) for k, v in metrics_point.items()},
        'conformal': conformal_results,
        'strata': {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                       for kk, vv in v.items()} for k, v in strata_results.items()},
        'DSCE_duration': float(dsce),
        'MCE_duration': float(mce),
        'DSCE_progress': float(dsce_progress),
        'MCE_progress': float(mce_progress),
        'uncertainty_error_spearman': float(rho),
        'ensemble_k': args.top_k,
        'test_samples': N,
        'mean_sigma': float(y_std.mean()),
        'median_sigma': float(np.median(y_std)),
    }

    results_path = output_dir / 'uncertainty_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save predictions with uncertainty for potential further analysis
    np.savez(output_dir / 'predictions_with_uncertainty.npz',
             y_pred=y_pred, y_true=y_true, y_std=y_std,
             y_lower_90=y_lower_full, y_upper_90=y_upper_full)
    print(f"Predictions saved to {output_dir / 'predictions_with_uncertainty.npz'}")

    notify_discord(f"✅ Uncertainty quantification complete!\n"
                   f"MAE={metrics_point['MAE']:.2f}h, MedAE={metrics_point['MedAE']:.2f}h, R²={metrics_point['R2']:.4f}\n"
                   f"90% PICP(adaptive)={conformal_results['90']['adaptive']['PICP']*100:.1f}%, "
                   f"MPIW={conformal_results['90']['adaptive']['MPIW']:.1f}h\n"
                   f"DSCE={dsce*100:.2f}pp, ρ(σ,|e|)={rho:.3f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
