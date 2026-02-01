"""
港口停靠时间预测模型
使用简单的神经网络预测船舶在港口的停靠时间
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict
import json


class PortStopDataset(Dataset):
    """港口停靠数据集"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class PortStopPredictor(nn.Module):
    """
    港口停靠时间预测模型
    
    输入特征:
        - 港口位置 (lon, lat)
        - 港口区域 (one-hot编码)
        - 月份 (周期编码)
        - 星期几 (周期编码)
    
    输出:
        - 预测停靠时间 (小时)
    """
    
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
        layers.append(nn.ReLU())  # 停靠时间必须为正
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class PortStopModel:
    """港口停靠时间预测模型管理器"""
    
    def __init__(self, model_dir: str = './output/port_model'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model: Optional[PortStopPredictor] = None
        self.scaler = StandardScaler()
        self.region_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 特征列
        self.feature_cols = ['lon', 'lat', 'month_sin', 'month_cos', 
                            'weekday_sin', 'weekday_cos']
        self.region_classes = []
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """准备特征"""
        features = df.copy()
        
        # 时间特征
        features['arrival_time'] = pd.to_datetime(features['arrival_time'])
        features['month'] = features['arrival_time'].dt.month
        features['weekday'] = features['arrival_time'].dt.weekday
        
        # 周期编码
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['weekday_sin'] = np.sin(2 * np.pi * features['weekday'] / 7)
        features['weekday_cos'] = np.cos(2 * np.pi * features['weekday'] / 7)
        
        # 区域编码
        if fit:
            self.region_encoder.fit(features['region'])
            self.region_classes = list(self.region_encoder.classes_)
        
        region_encoded = self.region_encoder.transform(features['region'])
        num_regions = len(self.region_classes)
        region_onehot = np.zeros((len(features), num_regions))
        region_onehot[np.arange(len(features)), region_encoded] = 1
        
        # 组合特征
        numeric_features = features[self.feature_cols].values
        all_features = np.hstack([numeric_features, region_onehot])
        
        # 标准化
        if fit:
            all_features = self.scaler.fit_transform(all_features)
        else:
            all_features = self.scaler.transform(all_features)
        
        return all_features
    
    def train(self, stop_df: pd.DataFrame, epochs: int = 100, 
              batch_size: int = 32, lr: float = 0.001) -> Dict:
        """训练模型"""
        print(f"Training port stop predictor on {len(stop_df)} samples...")
        
        # 过滤异常值
        stop_df = stop_df[stop_df['duration_hours'] > 0].copy()
        stop_df = stop_df[stop_df['duration_hours'] < 200]  # 最多200小时
        
        if len(stop_df) < 10:
            raise ValueError("Not enough data to train")
        
        # 准备特征和目标
        X = self.prepare_features(stop_df, fit=True)
        y = np.log1p(stop_df['duration_hours'].values)  # log变换使分布更正态
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 创建数据加载器
        train_dataset = PortStopDataset(X_train, y_train)
        val_dataset = PortStopDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 初始化模型
        input_dim = X.shape[1]
        self.model = PortStopPredictor(input_dim).to(self.device)
        
        # 训练
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.HuberLoss()
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    pred = self.model(batch_x)
                    val_loss += criterion(pred, batch_y).item()
            
            val_loss /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # 计算最终指标
        self.model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            pred_log = self.model(X_val_tensor).cpu().numpy()
            pred_hours = np.expm1(pred_log)  # 逆log变换
            actual_hours = np.expm1(y_val)
            
            mae = np.mean(np.abs(pred_hours - actual_hours))
            rmse = np.sqrt(np.mean((pred_hours - actual_hours)**2))
        
        print(f"\nFinal Metrics:")
        print(f"  MAE: {mae:.2f} hours")
        print(f"  RMSE: {rmse:.2f} hours")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'history': history
        }
    
    def predict(self, stop_df: pd.DataFrame) -> np.ndarray:
        """预测停靠时间"""
        if self.model is None:
            self.load()
        
        X = self.prepare_features(stop_df, fit=False)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            pred_log = self.model(X_tensor).cpu().numpy()
            pred_hours = np.expm1(pred_log)
        
        return pred_hours
    
    def predict_single(self, lon: float, lat: float, region: str,
                       arrival_time: pd.Timestamp) -> float:
        """预测单个停靠的时间"""
        df = pd.DataFrame([{
            'lon': lon,
            'lat': lat,
            'region': region,
            'arrival_time': arrival_time
        }])
        return self.predict(df)[0]
    
    def save(self):
        """保存模型"""
        if self.model is None:
            return
        
        # 保存模型权重
        torch.save(self.model.state_dict(), self.model_dir / 'model.pth')
        
        # 保存scaler和encoder
        np.save(self.model_dir / 'scaler_mean.npy', self.scaler.mean_)
        np.save(self.model_dir / 'scaler_scale.npy', self.scaler.scale_)
        
        # 保存配置
        config = {
            'region_classes': self.region_classes,
            'feature_cols': self.feature_cols,
            'input_dim': self.model.network[0].in_features
        }
        with open(self.model_dir / 'config.json', 'w') as f:
            json.dump(config, f)
        
        print(f"Model saved to {self.model_dir}")
    
    def load(self):
        """加载模型"""
        # 加载配置
        with open(self.model_dir / 'config.json', 'r') as f:
            config = json.load(f)
        
        self.region_classes = config['region_classes']
        self.feature_cols = config['feature_cols']
        self.region_encoder.classes_ = np.array(self.region_classes)
        
        # 加载scaler
        self.scaler.mean_ = np.load(self.model_dir / 'scaler_mean.npy')
        self.scaler.scale_ = np.load(self.model_dir / 'scaler_scale.npy')
        
        # 加载模型
        self.model = PortStopPredictor(config['input_dim']).to(self.device)
        self.model.load_state_dict(torch.load(self.model_dir / 'model.pth', 
                                               map_location=self.device))
        
        print(f"Model loaded from {self.model_dir}")


def get_average_stop_time_by_region(stop_df: pd.DataFrame) -> Dict[str, float]:
    """计算每个区域的平均停靠时间（作为简单baseline）"""
    return stop_df.groupby('region')['duration_hours'].mean().to_dict()


if __name__ == '__main__':
    # 测试
    stop_df = pd.read_csv('./output/processed/port_stops.csv')
    
    if len(stop_df) > 0:
        print("Stop data loaded:", len(stop_df), "records")
        print("\nRegion distribution:")
        print(stop_df['region'].value_counts())
        
        # 简单baseline
        print("\nAverage stop time by region:")
        avg_by_region = get_average_stop_time_by_region(stop_df)
        for region, avg_time in avg_by_region.items():
            print(f"  {region}: {avg_time:.2f} hours")
        
        # 训练模型
        model = PortStopModel()
        metrics = model.train(stop_df, epochs=100)
