#!/usr/bin/env python
"""
Reduce large deviations in ETA predictions via three strategies:

1. Post-hoc isotonic calibration (no retraining)
2. Inverse-density sample weighting (training modification)
3. Dual-space loss: log-space + hours-space MSE (training modification)

Strategy 1 can be evaluated immediately on existing predictions.
Strategies 2 & 3 require retraining.

Usage:
    # Evaluate calibration (no retraining)
    python reduce_large_deviations.py --strategy calibrate --pred_dir output/mstgn_mlp2_kd06

    # Train with sample weighting
    python reduce_large_deviations.py --strategy weight --output_dir output/mstgn_kd06_weighted

    # Train with dual-space loss
    python reduce_large_deviations.py --strategy dual_loss --output_dir output/mstgn_kd06_dualspace
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Re-use helpers from training
sys.path.insert(0, os.path.dirname(__file__))


def inverse_normalize_target(normalized, target_mean, target_std):
    log_target = normalized * target_std + target_mean
    return np.expm1(log_target)


def calculate_metrics(y_pred, y_true, label=""):
    abs_err = np.abs(y_pred - y_true)
    error = y_pred - y_true
    mae = abs_err.mean()
    rmse = np.sqrt((error**2).mean())
    mask = y_true > 24
    mape = np.mean(np.abs(error[mask]) / y_true[mask]) * 100 if mask.sum() > 0 else 0
    large_100 = (abs_err > 100).sum()
    large_100_rate = large_100 / len(y_true) * 100
    # Per-bin
    results = {
        'MAE': mae, 'RMSE': rmse, 'MAPE': mape,
        'large_100h': large_100, 'large_100h_rate': large_100_rate,
        'P99': np.percentile(abs_err, 99),
    }
    # Per-duration-bin
    bins = [(0, 168), (168, 336), (336, 504), (504, 720)]
    for lo, hi in bins:
        m = (y_true >= lo) & (y_true < hi)
        if m.sum() > 0:
            ae_bin = abs_err[m]
            results[f'MAE_{lo}_{hi}'] = float(ae_bin.mean())
            results[f'large_{lo}_{hi}'] = int((ae_bin > 100).sum())
            results[f'rate_{lo}_{hi}'] = float((ae_bin > 100).sum() / m.sum() * 100)

    if label:
        print(f"\n--- {label} ---")
    print(f"  MAE: {mae:.2f}h | RMSE: {rmse:.2f}h | MAPE: {mape:.2f}%")
    print(f"  |err|>100h: {large_100:,} ({large_100_rate:.3f}%) | P99: {results['P99']:.1f}h")
    for lo, hi in bins:
        key_mae = f'MAE_{lo}_{hi}'
        key_rate = f'rate_{lo}_{hi}'
        if key_mae in results:
            m = (y_true >= lo) & (y_true < hi)
            print(f"  [{lo},{hi})h: n={m.sum():,}, MAE={results[key_mae]:.2f}h, "
                  f"|err|>100h={results.get(f'large_{lo}_{hi}', 0):,} "
                  f"({results[key_rate]:.3f}%)")
    return results


# ============================================================
# Strategy 1: Post-hoc calibration
# ============================================================

def strategy_calibrate(pred_dir, norm_path):
    """Apply post-hoc calibration to fix systematic bias in long-voyage predictions."""
    from sklearn.isotonic import IsotonicRegression

    norm = np.load(norm_path, allow_pickle=True)
    target_mean, target_std = float(norm['target_mean']), float(norm['target_std'])

    d = np.load(Path(pred_dir) / 'predictions.npz')
    y_pred, y_true = d['y_pred'], d['y_true']

    print("=" * 70)
    print("STRATEGY 1: POST-HOC CALIBRATION")
    print("=" * 70)

    calculate_metrics(y_pred, y_true, "Baseline (uncalibrated)")

    # Load validation predictions for fitting calibration
    # We need to re-evaluate on validation set to get val predictions
    # For now, use a held-out portion of test as a proxy (first 30% as "cal", rest as "eval")
    # Better: use validation predictions if available
    val_pred_path = Path(pred_dir) / 'val_predictions.npz'
    if val_pred_path.exists():
        val_d = np.load(val_pred_path)
        y_pred_cal, y_true_cal = val_d['y_pred'], val_d['y_true']
        y_pred_eval, y_true_eval = y_pred, y_true
        print("  Using validation set for calibration")
    else:
        # Split test set: calibration (30%) + evaluation (70%)
        n = len(y_pred)
        rng = np.random.RandomState(42)
        idx = rng.permutation(n)
        cal_n = int(0.3 * n)
        cal_idx, eval_idx = idx[:cal_n], idx[cal_n:]
        y_pred_cal, y_true_cal = y_pred[cal_idx], y_true[cal_idx]
        y_pred_eval, y_true_eval = y_pred[eval_idx], y_true[eval_idx]
        print(f"  Using test split: cal={cal_n:,}, eval={n-cal_n:,}")

    # Method 1a: Global isotonic regression
    print("\n  Fitting global isotonic regression...")
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_pred_cal, y_true_cal)
    y_cal_iso = np.maximum(iso.predict(y_pred_eval), 0)
    calculate_metrics(y_cal_iso, y_true_eval, "Global Isotonic Regression")

    # Method 1b: Piecewise linear correction for long predictions
    print("\n  Fitting piecewise linear correction (long voyages)...")
    # Fit correction only for predictions > 168h
    high_mask_cal = y_pred_cal > 168
    if high_mask_cal.sum() > 100:
        from numpy.polynomial import polynomial as P
        # Fit linear correction: y_true = a*y_pred + b for high predictions
        coeffs = np.polyfit(y_pred_cal[high_mask_cal], y_true_cal[high_mask_cal], 1)
        a, b = coeffs
        print(f"    Linear fit (pred>168h): y_corrected = {a:.4f}*y_pred + {b:.2f}")

        y_cal_pw = y_pred_eval.copy()
        high_mask_eval = y_pred_eval > 168
        y_cal_pw[high_mask_eval] = a * y_pred_eval[high_mask_eval] + b
        y_cal_pw = np.maximum(y_cal_pw, 0)
        calculate_metrics(y_cal_pw, y_true_eval, "Piecewise Linear (pred>168h)")

    # Method 1c: Quantile-based correction
    # For each prediction quantile, compute the empirical bias and correct
    print("\n  Fitting quantile-based correction...")
    n_bins = 50
    quantiles = np.percentile(y_pred_cal, np.linspace(0, 100, n_bins + 1))
    corrections = np.zeros(n_bins)
    for i in range(n_bins):
        mask_bin = (y_pred_cal >= quantiles[i]) & (y_pred_cal < quantiles[i+1])
        if mask_bin.sum() > 10:
            corrections[i] = np.mean(y_true_cal[mask_bin] - y_pred_cal[mask_bin])

    # Apply corrections
    y_cal_qb = y_pred_eval.copy()
    for i in range(n_bins):
        mask_bin = (y_pred_eval >= quantiles[i]) & (y_pred_eval < quantiles[i+1])
        y_cal_qb[mask_bin] += corrections[i]
    y_cal_qb = np.maximum(y_cal_qb, 0)
    calculate_metrics(y_cal_qb, y_true_eval, "Quantile-Binned Correction (50 bins)")

    # Method 1d: Simple additive correction per duration bin
    print("\n  Fitting duration-bin correction...")
    pred_bins = [(0, 72), (72, 168), (168, 336), (336, 504), (504, 800)]
    bin_offsets = {}
    for lo, hi in pred_bins:
        mask = (y_pred_cal >= lo) & (y_pred_cal < hi)
        if mask.sum() > 10:
            bias = np.mean(y_true_cal[mask] - y_pred_cal[mask])
            bin_offsets[(lo, hi)] = bias
            print(f"    pred [{lo},{hi})h: bias={bias:+.2f}h (n={mask.sum():,})")

    y_cal_db = y_pred_eval.copy()
    for (lo, hi), offset in bin_offsets.items():
        mask = (y_pred_eval >= lo) & (y_pred_eval < hi)
        y_cal_db[mask] += offset
    y_cal_db = np.maximum(y_cal_db, 0)
    calculate_metrics(y_cal_db, y_true_eval, "Duration-Bin Additive Correction")

    return y_cal_iso, y_true_eval


# ============================================================
# Strategy 2: Sample weighting (training modification)
# ============================================================

def compute_sample_weights(y_train, norm_path, method='sqrt_inv_density'):
    """Compute per-sample weights for imbalanced regression.

    Based on: Steininger et al. (2021) "Density-based weighting for imbalanced regression"
    """
    norm = np.load(norm_path, allow_pickle=True)
    target_mean, target_std = float(norm['target_mean']), float(norm['target_std'])

    # Convert to hours for binning
    y_hours = inverse_normalize_target(y_train, target_mean, target_std)

    if method == 'sqrt_inv_density':
        # Bin by duration, weight by sqrt(1/count) to up-weight rare bins
        bin_edges = np.array([0, 24, 72, 168, 336, 504, 800])
        bin_idx = np.digitize(y_hours, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
        bin_counts = np.bincount(bin_idx, minlength=len(bin_edges) - 1)
        bin_weights = 1.0 / np.sqrt(np.maximum(bin_counts, 1))
        # Normalize so mean weight = 1
        sample_weights = bin_weights[bin_idx]
        sample_weights = sample_weights / sample_weights.mean()
    elif method == 'lds':
        # Label Distribution Smoothing (Yang et al., 2021)
        from scipy.ndimage import gaussian_filter1d
        # Create histogram of targets
        n_bins = 100
        hist, edges = np.histogram(y_hours, bins=n_bins, range=(0, 720))
        # Smooth with Gaussian kernel
        smoothed = gaussian_filter1d(hist.astype(float), sigma=2)
        smoothed = np.maximum(smoothed, 1)
        # Effective density per sample
        bin_idx = np.clip(np.digitize(y_hours, edges) - 1, 0, n_bins - 1)
        density = smoothed[bin_idx]
        sample_weights = 1.0 / np.sqrt(density)
        sample_weights = sample_weights / sample_weights.mean()
    else:
        sample_weights = np.ones(len(y_train))

    return sample_weights.astype(np.float32)


# ============================================================
# Strategy 3: Dual-space loss
# ============================================================

class DualSpaceLoss:
    """Combined loss: log-space MSE + hours-space MAE.

    L = (1-β) * MSE(ŷ_norm, y_norm) + β * MAE(ŷ_hours, y_hours)

    With β small (e.g., 0.1), the hours-space term acts as regularization
    that prevents the model from ignoring large absolute errors
    that are compressed in log-space.
    """
    def __init__(self, target_mean, target_std, beta=0.1):
        import torch
        self.target_mean = target_mean
        self.target_std = target_std
        self.beta = beta

    def __call__(self, pred_norm, y_norm):
        import torch
        loss_log = torch.nn.functional.mse_loss(pred_norm, y_norm)

        if self.beta > 0:
            # Convert to hours for the secondary loss
            pred_hours = torch.expm1(pred_norm * self.target_std + self.target_mean)
            y_hours = torch.expm1(y_norm * self.target_std + self.target_mean)
            loss_hours = torch.mean(torch.abs(pred_hours - y_hours))
            # Normalize hours loss to similar scale as log loss
            loss_hours_scaled = loss_hours / 100.0  # Rough scaling factor
            return (1 - self.beta) * loss_log + self.beta * loss_hours_scaled
        return loss_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, required=True,
                        choices=['calibrate', 'weight', 'dual_loss', 'all'],
                        help='Which strategy to evaluate/train')
    parser.add_argument('--pred_dir', type=str, default='output/mstgn_mlp2_kd06',
                        help='Prediction directory (for calibrate)')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--norm_path', type=str, default='output/norm_params.npz')
    parser.add_argument('--cache_dir', type=str,
                        default='output/cache_sequences/seq48_label24_pred1_mv150000_ms50000000')
    parser.add_argument('--weight_method', type=str, default='sqrt_inv_density',
                        choices=['sqrt_inv_density', 'lds'])
    parser.add_argument('--dual_beta', type=float, default=0.1,
                        help='Weight for hours-space loss in dual-space loss')
    args = parser.parse_args()

    if args.strategy == 'calibrate':
        strategy_calibrate(args.pred_dir, args.norm_path)
    elif args.strategy == 'weight':
        # Compute and save sample weights for use in training
        y_train = np.load(Path(args.cache_dir) / 'y_train.npy', mmap_mode='r')
        counts = np.load(Path(args.cache_dir) / 'actual_counts.npy', allow_pickle=True).item()
        y_train = np.array(y_train[:counts['train']])
        print(f"Computing sample weights ({args.weight_method}) for {len(y_train):,} train samples...")
        weights = compute_sample_weights(y_train, args.norm_path, args.weight_method)
        out_path = Path(args.cache_dir) / f'sample_weights_{args.weight_method}.npy'
        np.save(out_path, weights)
        print(f"Saved to {out_path}")
        print(f"  Weight stats: mean={weights.mean():.4f}, min={weights.min():.4f}, "
              f"max={weights.max():.4f}, std={weights.std():.4f}")
        # Show weights per bin
        norm = np.load(args.norm_path, allow_pickle=True)
        y_hours = inverse_normalize_target(y_train,
                                           float(norm['target_mean']),
                                           float(norm['target_std']))
        for lo, hi in [(0, 24), (24, 72), (72, 168), (168, 336), (336, 504), (504, 720)]:
            m = (y_hours >= lo) & (y_hours < hi)
            if m.sum() > 0:
                print(f"  [{lo},{hi})h: n={m.sum():,}, mean_weight={weights[m].mean():.4f}")
    elif args.strategy == 'all':
        print("=" * 70)
        print("EVALUATING ALL STRATEGIES")
        print("=" * 70)
        strategy_calibrate(args.pred_dir, args.norm_path)


if __name__ == '__main__':
    main()
