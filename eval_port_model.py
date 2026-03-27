"""
Evaluate and improve port dwell time prediction.

Problems with the existing model:
1. Raw stops include berth changes (median inter-stop gap = 1.6h)
2. Only 12 weak features (location + time cyclical)
3. Tiny 2-layer MLP on highly variable target (std/mean = 125%)

Improvements:
1. Merge nearby stops into port visits (64.7% reduction)
2. Add vessel-specific historical features
3. Use XGBoost (better for tabular data with 14K samples)
4. Evaluate integration for multi-segment ETA

Usage:
    python eval_port_model.py                       # Full evaluation
    python eval_port_model.py --save_model           # Train and save best model
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Step 1: Merge berth changes into port visits
# ============================================================

def merge_port_stops(stops_df, gap_threshold_h=6, dist_threshold_deg=0.5):
    """Merge consecutive stops within gap_threshold hours and dist_threshold degrees
    into single port visits. This collapses berth changes / pilot movements."""
    stops_sorted = stops_df.sort_values(['mmsi', 'arrival_time']).copy()
    stops_sorted['arrival_time'] = pd.to_datetime(stops_sorted['arrival_time'])
    stops_sorted['departure_time'] = pd.to_datetime(stops_sorted['departure_time'])

    all_merged = []
    for mmsi, grp in stops_sorted.groupby('mmsi'):
        current = None
        for _, row in grp.iterrows():
            if current is None:
                current = dict(row)
                continue
            gap = (row['arrival_time'] - pd.Timestamp(current['departure_time'])).total_seconds() / 3600
            dist = ((row['lon'] - current['lon'])**2 + (row['lat'] - current['lat'])**2)**0.5
            if gap < gap_threshold_h and dist < dist_threshold_deg:
                current['departure_time'] = row['departure_time']
                current['duration_hours'] = (
                    pd.Timestamp(current['departure_time']) -
                    pd.Timestamp(current['arrival_time'])
                ).total_seconds() / 3600
            else:
                all_merged.append(current)
                current = dict(row)
        if current:
            all_merged.append(current)

    merged = pd.DataFrame(all_merged)
    merged['arrival_time'] = pd.to_datetime(merged['arrival_time'])
    merged['departure_time'] = pd.to_datetime(merged['departure_time'])
    return merged


# ============================================================
# Step 2: Feature engineering
# ============================================================

def engineer_features(df):
    """Create features for port dwell prediction."""
    df = df.sort_values(['mmsi', 'arrival_time']).copy()

    # Time features
    df['month'] = df['arrival_time'].dt.month
    df['weekday'] = df['arrival_time'].dt.weekday
    df['hour'] = df['arrival_time'].dt.hour
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['wd_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['wd_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Region encoding
    df['region_code'] = pd.Categorical(df['region']).codes

    # Vessel-specific history (expanding mean, shifted to avoid leakage)
    df['mmsi_hist_mean'] = (df.groupby('mmsi')['duration_hours']
                            .expanding().mean()
                            .reset_index(level=0, drop=True)
                            .shift(1))
    df['mmsi_hist_std'] = (df.groupby('mmsi')['duration_hours']
                           .expanding().std()
                           .reset_index(level=0, drop=True)
                           .shift(1))
    df['mmsi_visit_count'] = df.groupby('mmsi').cumcount()

    # Preceding sailing gap (time since last departure)
    df['prev_dep'] = df.groupby('mmsi')['departure_time'].shift(1)
    df['prev_sailing_hours'] = (df['arrival_time'] - df['prev_dep']).dt.total_seconds() / 3600

    # Region-level rolling stats (global, shifted)
    df['region_hist_mean'] = (df.groupby('region')['duration_hours']
                              .expanding().mean()
                              .reset_index(level=0, drop=True)
                              .shift(1))

    # Fill NaN for first visits
    global_median = df['duration_hours'].median()
    df['mmsi_hist_mean'] = df['mmsi_hist_mean'].fillna(global_median)
    df['mmsi_hist_std'] = df['mmsi_hist_std'].fillna(0)
    df['prev_sailing_hours'] = df['prev_sailing_hours'].fillna(0)
    df['region_hist_mean'] = df['region_hist_mean'].fillna(global_median)

    return df


FEATURE_COLS = [
    'lon', 'lat',
    'month_sin', 'month_cos', 'wd_sin', 'wd_cos', 'hour_sin', 'hour_cos',
    'region_code',
    'mmsi_hist_mean', 'mmsi_hist_std', 'mmsi_visit_count',
    'prev_sailing_hours',
    'region_hist_mean',
]


# ============================================================
# Step 3: Model training and evaluation
# ============================================================

def evaluate_baselines(df):
    """Evaluate simple baselines."""
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)

    y = df['duration_hours'].values

    # Global median
    global_median = np.median(y)
    mae_global = mean_absolute_error(y, np.full(len(y), global_median))
    print(f"Global median ({global_median:.1f}h):  MAE = {mae_global:.2f}h")

    # Region median
    rmed = df.groupby('region')['duration_hours'].median()
    pred_rmed = df['region'].map(rmed).values
    mae_rmed = mean_absolute_error(y, pred_rmed)
    print(f"Region median:                MAE = {mae_rmed:.2f}h")

    # Per-MMSI-region median (in-sample, optimistic)
    mmr = df.groupby(['mmsi', 'region'])['duration_hours'].transform('median')
    mae_mmr = mean_absolute_error(y, mmr)
    print(f"Per-MMSI-region median (IS):  MAE = {mae_mmr:.2f}h")

    return {'global_median': global_median, 'region_median': rmed}


def train_xgboost(df, feature_cols=FEATURE_COLS):
    """Train XGBoost on port dwell data with temporal split."""
    # Temporal split: use last 20% by time as test
    df = df.sort_values('arrival_time')
    n = len(df)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # Log-transform target
    y_train = np.log1p(train_df['duration_hours'].values)
    y_val = np.log1p(val_df['duration_hours'].values)
    y_test = np.log1p(test_df['duration_hours'].values)

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Predict on test
    pred_log = model.predict(X_test)
    pred_hours = np.expm1(pred_log)
    actual_hours = np.expm1(y_test)
    pred_hours = np.maximum(pred_hours, 0)

    mae = mean_absolute_error(actual_hours, pred_hours)
    rmse = np.sqrt(mean_squared_error(actual_hours, pred_hours))

    # Baseline comparison on test set
    test_median = np.full(len(actual_hours), np.median(np.expm1(y_train)))
    mae_baseline = mean_absolute_error(actual_hours, test_median)

    print(f"\n{'=' * 60}")
    print(f"XGBOOST EVALUATION (temporal split)")
    print(f"{'=' * 60}")
    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    print(f"  Train median: {np.median(np.expm1(y_train)):.1f}h")
    print(f"  Test  median: {np.median(actual_hours):.1f}h")
    print(f"  Baseline MAE (train median): {mae_baseline:.2f}h")
    print(f"  XGBoost MAE:  {mae:.2f}h")
    print(f"  XGBoost RMSE: {rmse:.2f}h")
    print(f"  Improvement over baseline: {(1 - mae/mae_baseline)*100:.1f}%")

    # Feature importance
    fi = dict(zip(feature_cols, model.feature_importances_))
    fi_sorted = sorted(fi.items(), key=lambda x: -x[1])
    print(f"\n  Feature importance:")
    for name, imp in fi_sorted:
        print(f"    {name:25s} {imp:.3f}")

    # Per-region performance
    print(f"\n  Per-region performance:")
    for region in test_df['region'].unique():
        mask = test_df['region'].values == region
        if mask.sum() > 10:
            r_mae = mean_absolute_error(actual_hours[mask], pred_hours[mask])
            r_mean = actual_hours[mask].mean()
            print(f"    {region:15s}: n={mask.sum():5d}, MAE={r_mae:.1f}h, mean_actual={r_mean:.1f}h")

    return model, test_df, pred_hours, actual_hours


# ============================================================
# Step 4: Multi-segment ETA evaluation
# ============================================================

def evaluate_multiseg_integration(stops_merged, voyages_dir, norm_path, pred_dir):
    """Evaluate whether adding port dwell improves total ETA for multi-segment voyages."""
    print(f"\n{'=' * 60}")
    print(f"MULTI-SEGMENT INTEGRATION EVALUATION")
    print(f"{'=' * 60}")

    # Load sailing predictions
    pred_path = Path(pred_dir) / 'predictions.npz'
    if not pred_path.exists():
        print(f"  No predictions at {pred_path}, skipping")
        return

    d = np.load(pred_path)
    y_pred_sailing = d['y_pred']
    y_true_sailing = d['y_true']

    # The sailing model predicts remaining hours of the CURRENT segment
    # For multi-segment voyages, the true total time includes port dwells

    # Since our test set only has per-segment predictions, we can only
    # evaluate port dwell as an ADDITIVE correction for voyages that
    # are part of multi-segment journeys.

    # Statistics
    ae_sailing = np.abs(y_pred_sailing - y_true_sailing)
    print(f"  Sailing-only MAE: {ae_sailing.mean():.2f}h")
    print(f"  Total test samples: {len(y_true_sailing):,}")

    # Port dwell statistics for context
    print(f"\n  Port visit statistics (merged):")
    print(f"    Total visits: {len(stops_merged):,}")
    print(f"    Mean dwell: {stops_merged['duration_hours'].mean():.1f}h")
    print(f"    Median dwell: {stops_merged['duration_hours'].median():.1f}h")
    print(f"    Std dwell: {stops_merged['duration_hours'].std():.1f}h")

    # For cross-region transitions (the real multi-segment cases)
    ss = stops_merged.sort_values(['mmsi', 'arrival_time'])
    ss['prev_region'] = ss.groupby('mmsi')['region'].shift(1)
    ss['prev_dep'] = ss.groupby('mmsi')['departure_time'].shift(1)
    ss['gap_h'] = (ss['arrival_time'] - ss['prev_dep']).dt.total_seconds() / 3600
    cross = ss[ss['prev_region'] != ss['region']].dropna(subset=['prev_region'])
    print(f"\n  Cross-region transitions: {len(cross)}")
    print(f"    Mean sailing gap: {cross['gap_h'].mean():.1f}h")
    print(f"    Mean dwell at destination: {cross['duration_hours'].mean():.1f}h")

    # Key insight: for a multi-segment voyage like China→US,
    # the sailing time prediction already captures the sailing portion.
    # Adding port dwell would only matter if we're predicting total
    # journey time to FINAL destination (not just end of current segment).
    print(f"\n  Integration assessment:")
    print(f"    Current model: predicts remaining hours of CURRENT sailing segment")
    print(f"    Port dwell: would add predicted dwell time to segment prediction")
    print(f"    But: test data y_true is per-SEGMENT remaining hours")
    print(f"    → Adding port dwell to per-segment predictions would HURT accuracy")
    print(f"    → Port dwell is only useful for total journey time estimation")
    print(f"    → This requires a different prediction target (not currently available)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stops_path', default='output/processed/port_stops.csv')
    parser.add_argument('--pred_dir', default='output/mstgn_mlp2_kd06')
    parser.add_argument('--norm_path', default='output/norm_params.npz')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the trained XGBoost model')
    parser.add_argument('--output_dir', default='output/port_model')
    args = parser.parse_args()

    # Load raw stops
    stops = pd.read_csv(args.stops_path)
    stops['arrival_time'] = pd.to_datetime(stops['arrival_time'])
    stops['departure_time'] = pd.to_datetime(stops['departure_time'])
    print(f"Raw port stops: {len(stops):,}")

    # Merge berth changes
    merged = merge_port_stops(stops)
    print(f"Merged port visits: {len(merged):,} ({1 - len(merged)/len(stops):.1%} reduction)")
    print(f"Duration: mean={merged['duration_hours'].mean():.1f}h, "
          f"median={merged['duration_hours'].median():.1f}h, "
          f"std={merged['duration_hours'].std():.1f}h")

    # Filter outliers (> 500h = 3 weeks — likely data errors or dry dock)
    n_before = len(merged)
    merged = merged[merged['duration_hours'] <= 500].copy()
    print(f"After filtering >500h: {len(merged):,} (removed {n_before - len(merged)})")

    # Engineer features
    merged = engineer_features(merged)

    # Baselines
    evaluate_baselines(merged)

    # XGBoost
    model, test_df, pred_hours, actual_hours = train_xgboost(merged)

    # Multi-segment evaluation
    evaluate_multiseg_integration(merged, None, args.norm_path, args.pred_dir)

    # Save model if requested
    if args.save_model:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model.save_model(str(out_dir / 'port_xgb.json'))

        # Save merged stops and metadata
        config = {
            'feature_cols': FEATURE_COLS,
            'n_train_samples': len(merged) - len(test_df),
            'n_test_samples': len(test_df),
            'mae': float(mean_absolute_error(actual_hours, pred_hours)),
            'regions': sorted(merged['region'].unique().tolist()),
            'global_median_h': float(merged['duration_hours'].median()),
        }
        with open(out_dir / 'port_model_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        merged.to_csv(out_dir / 'merged_port_visits.csv', index=False)
        print(f"\nModel saved to {out_dir}")

    # Discord notification
    try:
        import requests
        webhook = "https://discord.com/api/webhooks/1477180434474336266/HQSS_BKlo1Ib-rzwItazg8ay0To2l-GR1GprnTvlPVpLxCxD9aBd4iHNgX"
        mae = mean_absolute_error(actual_hours, pred_hours)
        msg = (f"⚓ Port Dwell Model Evaluation Done!\n"
               f"Merged visits: {len(merged):,}\n"
               f"XGBoost MAE: {mae:.2f}h\n"
               f"Mean dwell: {merged['duration_hours'].mean():.1f}h")
        requests.post(webhook, json={"content": msg}, timeout=5)
    except Exception:
        pass


if __name__ == '__main__':
    main()
