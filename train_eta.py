#!/usr/bin/env python
"""
ETA预测模型训练脚本

双模型架构：
1. Informer-TP: 预测纯航行时间（使用预处理后的航程数据）
2. MLP: 预测港口停靠时间

最终ETA = 预测航行时间 + 预测停靠时间

用法：
    # 完整训练（包括停靠时间模型）
    python train_eta.py --epochs 3 --train_port_model
    
    # 仅训练航程模型
    python train_eta.py --epochs 3
    
    # 使用缓存的预处理数据
    python train_eta.py --use_cache --epochs 5
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.informer.model import Informer


# ============================================================
# Part 1: 港口停靠时间预测模型 (MLP)
# ============================================================

class PortStopPredictor(nn.Module):
    """港口停靠时间预测MLP"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class PortStopModel:
    """港口停靠时间预测模型管理器"""
    
    def __init__(self, model_dir: str = './output/port_model'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.region_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.region_classes = []
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """准备特征"""
        features = df.copy()
        
        features['arrival_time'] = pd.to_datetime(features['arrival_time'])
        features['month'] = features['arrival_time'].dt.month
        features['weekday'] = features['arrival_time'].dt.weekday
        
        # 周期编码
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['weekday_sin'] = np.sin(2 * np.pi * features['weekday'] / 7)
        features['weekday_cos'] = np.cos(2 * np.pi * features['weekday'] / 7)
        
        feature_cols = ['lon', 'lat', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos']
        
        # 区域编码
        if fit:
            self.region_encoder.fit(features['region'])
            self.region_classes = list(self.region_encoder.classes_)
        
        region_encoded = self.region_encoder.transform(features['region'])
        num_regions = len(self.region_classes)
        region_onehot = np.zeros((len(features), num_regions))
        region_onehot[np.arange(len(features)), region_encoded] = 1
        
        numeric_features = features[feature_cols].values
        all_features = np.hstack([numeric_features, region_onehot])
        
        if fit:
            all_features = self.scaler.fit_transform(all_features)
        else:
            all_features = self.scaler.transform(all_features)
        
        return all_features
    
    def train(self, stop_df: pd.DataFrame, epochs: int = 100, batch_size: int = 32, lr: float = 0.001):
        """训练模型"""
        print(f"Training port stop predictor on {len(stop_df)} samples...")
        
        stop_df = stop_df[stop_df['duration_hours'] > 0].copy()
        stop_df = stop_df[stop_df['duration_hours'] < 200]
        
        if len(stop_df) < 10:
            print("Not enough data to train port model")
            return None
        
        X = self.prepare_features(stop_df, fit=True)
        y = np.log1p(stop_df['duration_hours'].values)
        
        # 划分
        n = len(X)
        indices = np.random.permutation(n)
        train_size = int(n * 0.8)
        train_idx, val_idx = indices[:train_size], indices[train_size:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        input_dim = X.shape[1]
        self.model = PortStopPredictor(input_dim).to(self.device)
        
        optimizer = Adam(self.model.parameters(), lr=lr)
        criterion = nn.HuberLoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    val_loss += criterion(self.model(batch_x), batch_y).item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Val Loss={val_loss:.4f}")
        
        # 最终评估
        self.model.eval()
        with torch.no_grad():
            pred_log = self.model(torch.FloatTensor(X_val).to(self.device)).cpu().numpy()
            pred_hours = np.expm1(pred_log)
            actual_hours = np.expm1(y_val)
            mae = np.mean(np.abs(pred_hours - actual_hours))
        
        print(f"  Port model MAE: {mae:.2f} hours")
        return {'mae': mae}
    
    def predict(self, stop_df: pd.DataFrame) -> np.ndarray:
        """预测停靠时间"""
        if self.model is None:
            self.load()
        
        X = self.prepare_features(stop_df, fit=False)
        self.model.eval()
        with torch.no_grad():
            pred_log = self.model(torch.FloatTensor(X).to(self.device)).cpu().numpy()
        return np.expm1(pred_log)
    
    def save(self):
        if self.model is None:
            return
        torch.save(self.model.state_dict(), self.model_dir / 'model.pth')
        np.save(self.model_dir / 'scaler_mean.npy', self.scaler.mean_)
        np.save(self.model_dir / 'scaler_scale.npy', self.scaler.scale_)
        config = {'region_classes': self.region_classes, 'input_dim': self.model.network[0].in_features}
        with open(self.model_dir / 'config.json', 'w') as f:
            json.dump(config, f)
    
    def load(self):
        with open(self.model_dir / 'config.json', 'r') as f:
            config = json.load(f)
        self.region_classes = config['region_classes']
        self.region_encoder.classes_ = np.array(self.region_classes)
        self.scaler.mean_ = np.load(self.model_dir / 'scaler_mean.npy')
        self.scaler.scale_ = np.load(self.model_dir / 'scaler_scale.npy')
        self.model = PortStopPredictor(config['input_dim']).to(self.device)
        self.model.load_state_dict(torch.load(self.model_dir / 'model.pth', map_location=self.device))


# ============================================================
# Part 2: 航程ETA数据处理
# ============================================================

class VoyageETADataset:
    """航程数据处理器"""
    
    def __init__(self, seq_len: int = 48, label_len: int = 24, pred_len: int = 1):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        self.feature_min = None
        self.feature_max = None
        self.target_mean = None
        self.target_std = None
    
    def normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Minmax归一化"""
        if fit:
            self.feature_min = features.min(axis=0)
            self.feature_max = features.max(axis=0)
        
        range_vals = self.feature_max - self.feature_min
        range_vals[range_vals == 0] = 1.0
        return (features - self.feature_min) / range_vals
    
    def normalize_target(self, target: np.ndarray, fit: bool = False) -> np.ndarray:
        """Log + 标准化"""
        log_target = np.log1p(target)
        if fit:
            self.target_mean = log_target.mean()
            self.target_std = log_target.std()
        return (log_target - self.target_mean) / self.target_std
    
    def inverse_normalize_target(self, normalized: np.ndarray) -> np.ndarray:
        """反归一化"""
        log_target = normalized * self.target_std + self.target_mean
        return np.expm1(log_target)
    
    def create_sequences(self, voyage_df: pd.DataFrame) -> tuple:
        """从航程数据创建序列"""
        feature_cols = ['lat', 'lon', 'sog', 'cog']
        
        all_X = []
        all_X_mark = []
        all_y = []
        all_sailing_days = []
        
        for mmsi, group in voyage_df.groupby('mmsi'):
            group = group.sort_values('postime')
            
            features = group[feature_cols].values
            targets = group['remaining_hours'].values
            postime = pd.to_datetime(group['postime'])
            
            first_time = postime.iloc[0]
            sailing_days = (postime - first_time).dt.total_seconds() / 86400
            
            # 时间特征（5维）
            time_marks = np.stack([
                postime.dt.month.values / 12,
                postime.dt.day.values / 31,
                postime.dt.weekday.values / 7,
                postime.dt.hour.values / 24,
                postime.dt.minute.values / 60
            ], axis=1)
            
            # 滑动窗口
            for i in range(len(features) - self.seq_len - self.pred_len + 1):
                X = features[i:i+self.seq_len]
                X_mark_enc = time_marks[i:i+self.seq_len]
                X_mark_dec = time_marks[i+self.seq_len-self.label_len:i+self.seq_len+self.pred_len]
                y = targets[i+self.seq_len]
                sd = sailing_days.iloc[i+self.seq_len]
                
                all_X.append(X)
                all_X_mark.append((X_mark_enc, X_mark_dec))
                all_y.append(y)
                all_sailing_days.append(sd)
        
        X = np.array(all_X)
        X_mark_enc = np.array([m[0] for m in all_X_mark])
        X_mark_dec = np.array([m[1] for m in all_X_mark])
        y = np.array(all_y)
        sailing_days = np.array(all_sailing_days)
        
        # 归一化
        X_flat = X.reshape(-1, X.shape[-1])
        X_norm = self.normalize_features(X_flat, fit=True).reshape(X.shape)
        y_norm = self.normalize_target(y, fit=True)
        
        return X_norm, X_mark_enc, X_mark_dec, y_norm, sailing_days
    
    def save_params(self, path: str):
        """保存归一化参数"""
        np.savez(path,
                 feature_min=self.feature_min,
                 feature_max=self.feature_max,
                 target_mean=self.target_mean,
                 target_std=self.target_std)
    
    def load_params(self, path: str):
        """加载归一化参数"""
        data = np.load(path)
        self.feature_min = data['feature_min']
        self.feature_max = data['feature_max']
        self.target_mean = data['target_mean']
        self.target_std = data['target_std']


def create_data_loaders(X, X_mark_enc, X_mark_dec, y, sailing_days, label_len, pred_len, batch_size=32):
    """创建数据加载器"""
    n = len(X)
    indices = np.random.permutation(n)
    
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    
    dec_len = label_len + pred_len
    
    def make_loader(idx, shuffle=True):
        X_batch = torch.FloatTensor(X[idx])
        X_mark_enc_batch = torch.FloatTensor(X_mark_enc[idx])
        X_mark_dec_batch = torch.FloatTensor(X_mark_dec[idx])
        y_batch = torch.FloatTensor(y[idx])
        sd_batch = torch.FloatTensor(sailing_days[idx])
        
        X_dec = torch.zeros(len(idx), dec_len, X.shape[-1])
        X_dec[:, :label_len, :] = X_batch[:, -label_len:, :]
        
        dataset = TensorDataset(X_batch, X_mark_enc_batch, X_dec, X_mark_dec_batch, y_batch, sd_batch)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return (make_loader(train_idx, True), make_loader(val_idx, False), make_loader(test_idx, False),
            sailing_days[test_idx], y[test_idx], test_idx)


# ============================================================
# Part 3: Informer训练器
# ============================================================

class InformerTrainer:
    """Informer模型训练器"""
    
    def __init__(self, model, device, lr=5e-6):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
    
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(loader, desc="Training", leave=False):
            x_enc, x_mark_enc, x_dec, x_mark_dec, y, _ = batch
            
            x_enc = x_enc.to(self.device)
            x_mark_enc = x_mark_enc.to(self.device)
            x_dec = x_dec.to(self.device)
            x_mark_dec = x_mark_dec.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            output = output.squeeze(-1).squeeze(-1)
            
            loss = self.criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in loader:
                x_enc, x_mark_enc, x_dec, x_mark_dec, y, _ = batch
                
                x_enc = x_enc.to(self.device)
                x_mark_enc = x_mark_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                x_mark_dec = x_mark_dec.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                output = output.squeeze(-1).squeeze(-1)
                loss = self.criterion(output, y)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def predict(self, loader):
        self.model.eval()
        preds, trues, sailing_days_list = [], [], []
        
        with torch.no_grad():
            for batch in loader:
                x_enc, x_mark_enc, x_dec, x_mark_dec, y, sd = batch
                
                x_enc = x_enc.to(self.device)
                x_mark_enc = x_mark_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                x_mark_dec = x_mark_dec.to(self.device)
                
                output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                output = output.squeeze(-1).squeeze(-1)
                
                preds.append(output.cpu().numpy())
                trues.append(y.numpy())
                sailing_days_list.append(sd.numpy())
        
        return np.concatenate(preds), np.concatenate(trues), np.concatenate(sailing_days_list)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


# ============================================================
# Part 4: 评估和可视化
# ============================================================

def calculate_metrics(y_pred, y_true):
    """计算评估指标"""
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)
    
    mask = y_true > 24
    mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
    
    return {'MSE': mse, 'MAE_hours': mae, 'MAE_days': mae / 24, 'RMSE': rmse, 'MAPE': mape}


def plot_results(y_pred, y_true, sailing_days, save_dir):
    """绘制结果图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 误差vs航行天数
    ax = axes[0, 0]
    scatter = ax.scatter(sailing_days, np.abs(y_pred - y_true), c=y_true, cmap='viridis', alpha=0.5, s=10)
    ax.set_xlabel('Sailing Days')
    ax.set_ylabel('Absolute Error (hours)')
    ax.set_title('Prediction Error vs Sailing Days')
    plt.colorbar(scatter, ax=ax, label='True Remaining Hours')
    
    # 2. 分箱MAE
    ax = axes[0, 1]
    bins = np.arange(0, sailing_days.max() + 1, 1)
    bin_indices = np.digitize(sailing_days, bins)
    bin_maes, bin_centers = [], []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() > 10:
            bin_maes.append(np.mean(np.abs(y_pred[mask] - y_true[mask])))
            bin_centers.append((bins[i-1] + bins[i]) / 2)
    ax.bar(bin_centers, bin_maes, width=0.8, alpha=0.7, color='steelblue')
    ax.set_xlabel('Sailing Days')
    ax.set_ylabel('MAE (hours)')
    ax.set_title('MAE by Sailing Days')
    ax.grid(True, alpha=0.3)
    
    # 3. 预测vs实际
    ax = axes[1, 0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect')
    ax.set_xlabel('Actual Hours')
    ax.set_ylabel('Predicted Hours')
    ax.set_title('Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 误差分布
    ax = axes[1, 1]
    errors = y_pred - y_true
    ax.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--')
    ax.axvline(x=np.mean(errors), color='orange', linestyle='-', label=f'Mean: {np.mean(errors):.1f}h')
    ax.set_xlabel('Prediction Error (hours)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'analysis_plots.png'), dpi=150)
    plt.close()
    print(f"Plots saved to {save_dir}")


# ============================================================
# Part 5: 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='ETA预测模型训练')
    
    # 数据路径
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--processed_dir', type=str, default='./output/processed')
    
    # 模型参数
    parser.add_argument('--seq_len', type=int, default=48)
    parser.add_argument('--label_len', type=int, default=24)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.05)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-6)
    
    # 港口模型
    parser.add_argument('--train_port_model', action='store_true', help='训练港口停靠时间模型')
    parser.add_argument('--port_epochs', type=int, default=100)
    
    # 其他
    parser.add_argument('--use_cache', action='store_true', help='使用缓存的预处理数据')
    parser.add_argument('--max_files', type=int, default=None, help='处理文件数限制')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ===== Step 1: 加载数据 =====
    print("\n" + "="*60)
    print("Step 1: 加载数据")
    print("="*60)
    
    voyage_path = os.path.join(args.processed_dir, 'processed_voyages.csv')
    stop_path = os.path.join(args.processed_dir, 'port_stops.csv')
    
    if not os.path.exists(voyage_path):
        print("预处理数据不存在，请先运行: python preprocess_data.py")
        return
    
    voyage_df = pd.read_csv(voyage_path)
    stop_df = pd.read_csv(stop_path) if os.path.exists(stop_path) else pd.DataFrame()
    
    print(f"航程数据: {len(voyage_df):,} 条")
    print(f"停靠数据: {len(stop_df):,} 条")
    
    # ===== Step 2: 训练港口停靠模型（可选）=====
    if args.train_port_model and len(stop_df) > 10:
        print("\n" + "="*60)
        print("Step 2: 训练港口停靠时间预测模型")
        print("="*60)
        
        port_model = PortStopModel(os.path.join(args.output_dir, 'port_model'))
        port_model.train(stop_df, epochs=args.port_epochs)
        
        print("\n各区域平均停靠时间（baseline）:")
        for region, avg in stop_df.groupby('region')['duration_hours'].mean().items():
            print(f"  {region}: {avg:.2f} 小时")
    
    # ===== Step 3: 准备训练数据 =====
    print("\n" + "="*60)
    print("Step 3: 准备训练数据")
    print("="*60)
    
    # 过滤数据
    training_df = voyage_df[['lat', 'lon', 'sog', 'cog', 'remaining_hours', 'mmsi', 'postime']].copy()
    training_df = training_df.dropna()
    training_df = training_df[(training_df['remaining_hours'] >= 0) & (training_df['remaining_hours'] <= 720)]
    training_df = training_df[(training_df['sog'] >= 0) & (training_df['sog'] <= 30)]
    
    print(f"过滤后数据: {len(training_df):,} 条")
    
    if len(training_df) < 1000:
        print("数据量不足，请先处理更多文件")
        return
    
    dataset = VoyageETADataset(seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len)
    X, X_mark_enc, X_mark_dec, y, sailing_days = dataset.create_sequences(training_df)
    
    print(f"序列数量: {len(X):,}")
    print(f"序列形状: X={X.shape}, y={y.shape}")
    
    # 保存归一化参数
    dataset.save_params(os.path.join(args.output_dir, 'norm_params.npz'))
    
    train_loader, val_loader, test_loader, test_sd, test_y, test_idx = create_data_loaders(
        X, X_mark_enc, X_mark_dec, y, sailing_days,
        label_len=args.label_len, pred_len=args.pred_len, batch_size=args.batch_size
    )
    
    print(f"训练集: {len(train_loader.dataset):,}, 验证集: {len(val_loader.dataset):,}, 测试集: {len(test_loader.dataset):,}")
    
    # ===== Step 4: 训练Informer =====
    print("\n" + "="*60)
    print("Step 4: 训练Informer模型")
    print("="*60)
    
    model = Informer(
        enc_in=4, dec_in=4, c_out=1,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        d_model=args.d_model, n_heads=args.n_heads,
        e_layers=args.e_layers, d_layers=args.d_layers,
        d_ff=args.d_ff, dropout=args.dropout,
        attn='prob', embed='timeF', activation='gelu',
        output_attention=False, distil=True, device=device
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = InformerTrainer(model, device, lr=args.lr)
    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        trainer.scheduler.step(val_loss)
        
        lr = trainer.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={lr:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save(os.path.join(args.output_dir, 'best_informer.pth'))
            print("  -> 保存最佳模型!")
    
    trainer.load(os.path.join(args.output_dir, 'best_informer.pth'))
    
    # ===== Step 5: 评估 =====
    print("\n" + "="*60)
    print("Step 5: 评估")
    print("="*60)
    
    y_pred_norm, y_true_norm, pred_sd = trainer.predict(test_loader)
    
    y_pred = dataset.inverse_normalize_target(y_pred_norm)
    y_true = dataset.inverse_normalize_target(y_true_norm)
    y_pred = np.maximum(y_pred, 0)
    
    print(f"预测范围: [{y_pred.min():.2f}, {y_pred.max():.2f}] 小时")
    print(f"实际范围: [{y_true.min():.2f}, {y_true.max():.2f}] 小时")
    
    metrics = calculate_metrics(y_pred, y_true)
    print("\n【航程预测指标】")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # 保存结果
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"训练时间: {datetime.now().isoformat()}\n\n")
        f.write("配置:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
        f.write("\n航程ETA指标:\n")
        for name, value in metrics.items():
            f.write(f"  {name}: {value:.4f}\n")
    
    plot_results(y_pred, y_true, pred_sd, args.output_dir)
    
    print(f"\n训练完成! 结果保存至 {args.output_dir}")
    print("\n【最终ETA计算方式】")
    print("  最终ETA = Informer预测的航行时间 + 港口停靠时间模型预测的停靠时间")


if __name__ == '__main__':
    main()
