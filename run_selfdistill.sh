#!/bin/bash
# Refined experiments: checkpoint averaging + iterative distillation
set -e
cd /root/autodl-tmp/Multi-Model-ETA-Prediction
PY=/root/miniconda3/bin/python
CACHE=output/cache_sequences/seq48_label24_pred1_mv150000_ms50000000

echo "============================================"
echo "Exp 1: Ensemble KD α=0.6 + ckpt_avg=3"
echo "============================================"
$PY train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_kd06_avg3 \
    --distill --distill_alpha 0.6 \
    --ckpt_avg 3 \
    --seed 42 --epochs 15

echo ""
echo "============================================"
echo "Exp 2: Ensemble KD α=0.5 + ckpt_avg=3"
echo "============================================"
$PY train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_kd05_avg3 \
    --distill --distill_alpha 0.5 \
    --ckpt_avg 3 \
    --seed 42 --epochs 15

echo ""
echo "============================================"
echo "Exp 3: Ensemble KD α=0.6, more epochs + patience"
echo "============================================"
$PY train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_kd06_long \
    --distill --distill_alpha 0.6 \
    --seed 42 --epochs 25 --patience 6

echo ""
echo "============================================"
echo "Exp 4: Iterative distillation - use KD06 (15.13h) as teacher"
echo "============================================"
# Generate soft targets from the KD α=0.6 model (our best: 15.13h)
$PY generate_soft_targets.py \
    --teacher_ckpt output/mstgn_mlp2_kd06/best_mstgn.pth \
    --output_subdir soft_targets_iter1

# Train student from these improved soft targets
$PY train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_iter1_a05 \
    --distill --distill_alpha 0.5 \
    --soft_targets_dir $CACHE/soft_targets_iter1 \
    --seed 42 --epochs 15

echo ""
echo "============================================"
echo "Exp 5: Iterative distillation α=0.6"
echo "============================================"
$PY train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_iter1_a06 \
    --distill --distill_alpha 0.6 \
    --soft_targets_dir $CACHE/soft_targets_iter1 \
    --seed 42 --epochs 15

echo ""
echo "============================================"
echo "All experiments complete!"
echo "============================================"
