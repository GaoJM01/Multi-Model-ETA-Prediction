# 船舶ETA预测

```sh
# 数据预处理（提取最长航程、识别停靠）
python preprocess_data.py --data_dir ./data --output_dir ./output/processed

# 训练双模型
python train_eta.py --epochs 3 --train_port_model
```
