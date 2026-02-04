# 船舶ETA预测

```sh
# 数据预处理（提取最长航程、识别停靠）
python preprocess_data.py --data_dir ./data --output_dir ./output/processed

# 训练双模型
python train_eta.py --scheduler onecycle --epochs 5 --batch_size 256 --lr 3e-4 --seq_len 96 --train_port_model

# 从上次中断的地方继续训练
python train_eta.py --resume --epochs 10
```
