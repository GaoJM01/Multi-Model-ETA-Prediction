#!/bin/bash
set -e
cd /root/autodl-tmp/Multi-Model-ETA-Prediction

PYTHON=/root/miniconda3/bin/python
LOG=output/large_dev_experiments.log
CACHE=output/cache_sequences/seq48_label24_pred1_mv150000_ms50000000

echo "===============================================" | tee -a $LOG
echo "Large Deviation Reduction Experiments" | tee -a $LOG
echo "$(date)" | tee -a $LOG
echo "===============================================" | tee -a $LOG

# ============================================================
# Exp 1: Generate sample weights
# ============================================================
echo "" | tee -a $LOG
echo "=== Generating sample weights ===" | tee -a $LOG
$PYTHON reduce_large_deviations.py --strategy weight --weight_method sqrt_inv_density 2>&1 | tee -a $LOG
$PYTHON reduce_large_deviations.py --strategy weight --weight_method lds 2>&1 | tee -a $LOG

# ============================================================
# Exp 2: KD α=0.6 + sqrt_inv_density weighting
# ============================================================
echo "" | tee -a $LOG
echo "============================================" | tee -a $LOG
echo "Exp 2: KD α=0.6 + sqrt_inv_density weighting" | tee -a $LOG
echo "============================================" | tee -a $LOG
$PYTHON train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_kd06_weighted \
    --distill --distill_alpha 0.6 \
    --sample_weights ${CACHE}/sample_weights_sqrt_inv_density.npy \
    --seed 42 --epochs 15 --patience 4 2>&1 | tee -a $LOG

# ============================================================
# Exp 3: KD α=0.6 + LDS weighting
# ============================================================
echo "" | tee -a $LOG
echo "============================================" | tee -a $LOG
echo "Exp 3: KD α=0.6 + LDS weighting" | tee -a $LOG
echo "============================================" | tee -a $LOG
$PYTHON train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_kd06_lds \
    --distill --distill_alpha 0.6 \
    --sample_weights ${CACHE}/sample_weights_lds.npy \
    --seed 42 --epochs 15 --patience 4 2>&1 | tee -a $LOG

# ============================================================
# Exp 4: KD α=0.6 + dual-space loss β=0.1
# ============================================================
echo "" | tee -a $LOG
echo "============================================" | tee -a $LOG
echo "Exp 4: KD α=0.6 + dual-space loss β=0.1" | tee -a $LOG
echo "============================================" | tee -a $LOG
$PYTHON train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_kd06_dual01 \
    --distill --distill_alpha 0.6 \
    --dual_loss_beta 0.1 \
    --seed 42 --epochs 15 --patience 4 2>&1 | tee -a $LOG

# ============================================================
# Exp 5: KD α=0.6 + dual-space loss β=0.3
# ============================================================
echo "" | tee -a $LOG
echo "============================================" | tee -a $LOG
echo "Exp 5: KD α=0.6 + dual-space loss β=0.3" | tee -a $LOG
echo "============================================" | tee -a $LOG
$PYTHON train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_kd06_dual03 \
    --distill --distill_alpha 0.6 \
    --dual_loss_beta 0.3 \
    --seed 42 --epochs 15 --patience 4 2>&1 | tee -a $LOG

# ============================================================
# Exp 6: KD α=0.6 + weighting + dual-space (combined)
# ============================================================
echo "" | tee -a $LOG
echo "============================================" | tee -a $LOG
echo "Exp 6: KD α=0.6 + sqrt_inv_density + dual β=0.1" | tee -a $LOG
echo "============================================" | tee -a $LOG
$PYTHON train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_kd06_wd \
    --distill --distill_alpha 0.6 \
    --sample_weights ${CACHE}/sample_weights_sqrt_inv_density.npy \
    --dual_loss_beta 0.1 \
    --seed 42 --epochs 15 --patience 4 2>&1 | tee -a $LOG

# ============================================================
# Summary: Run analysis on all outputs
# ============================================================
echo "" | tee -a $LOG
echo "============================================" | tee -a $LOG
echo "Analysis Summary" | tee -a $LOG
echo "============================================" | tee -a $LOG
for dir in output/mstgn_kd06_weighted output/mstgn_kd06_lds output/mstgn_kd06_dual01 output/mstgn_kd06_dual03 output/mstgn_kd06_wd; do
    if [ -f "$dir/predictions.npz" ]; then
        echo "--- $dir ---" | tee -a $LOG
        $PYTHON -c "
import numpy as np
d = np.load('$dir/predictions.npz')
y_p, y_t = d['y_pred'], d['y_true']
ae = np.abs(y_p - y_t)
err = y_p - y_t
mae = ae.mean()
rmse = np.sqrt((err**2).mean())
l100 = (ae > 100).sum()
rate = l100 / len(y_t) * 100
# Per-bin
for lo, hi in [(168,336),(336,504),(504,720)]:
    m = (y_t >= lo) & (y_t < hi)
    if m.sum() > 0:
        be = ae[m]
        print(f'  [{lo},{hi})h: MAE={be.mean():.2f}h, |err|>100h={(be>100).sum()} ({(be>100).mean()*100:.2f}%)')
print(f'  TOTAL: MAE={mae:.2f}h, RMSE={rmse:.2f}h, |err|>100h={l100} ({rate:.3f}%)')
" 2>&1 | tee -a $LOG
    fi
done

echo "" | tee -a $LOG
echo "=== All experiments done ===" | tee -a $LOG
echo "$(date)" | tee -a $LOG
