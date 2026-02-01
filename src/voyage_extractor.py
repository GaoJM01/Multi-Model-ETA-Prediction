"""
航程提取器：从原始AIS数据中提取最长航段和港口停靠信息
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class VoyageSegment:
    """航段信息"""
    segment_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_hours: float
    start_lon: float
    end_lon: float
    start_lat: float
    end_lat: float
    num_points: int
    avg_sog: float
    data: pd.DataFrame  # 该航段的原始数据


@dataclass
class PortStop:
    """港口停靠信息"""
    port_id: int
    arrival_time: pd.Timestamp
    departure_time: pd.Timestamp
    duration_hours: float
    lon: float
    lat: float
    region: str  # 港口区域


class VoyageExtractor:
    """从AIS数据中提取航程信息"""
    
    # 速度阈值：低于此值视为停靠
    STOP_SPEED_THRESHOLD = 0.5  # 节
    
    # 最小航段时长（小时）：短于此值的航段不视为有效航段
    MIN_SEGMENT_HOURS = 6
    
    # 最小停靠时长（小时）：短于此值的停靠不视为港口停靠
    MIN_STOP_HOURS = 1
    
    # 港口区域定义（基于经度范围）
    PORT_REGIONS = {
        'china_east': {'lon_range': (100, 135), 'lat_range': (20, 45), 'name': '中国东部'},
        'singapore': {'lon_range': (103, 105), 'lat_range': (0, 3), 'name': '新加坡'},
        'middle_east': {'lon_range': (55, 80), 'lat_range': (10, 30), 'name': '中东/印度'},
        'suez': {'lon_range': (30, 55), 'lat_range': (25, 35), 'name': '红海/苏伊士'},
        'us_west': {'lon_range': (-130, -115), 'lat_range': (30, 50), 'name': '美国西海岸'},
        'pacific': {'lon_range': (-180, 180), 'lat_range': (-10, 60), 'name': '太平洋'},
    }
    
    def __init__(self, 
                 stop_speed_threshold: float = 0.5,
                 min_segment_hours: float = 6,
                 min_stop_hours: float = 1):
        self.stop_speed_threshold = stop_speed_threshold
        self.min_segment_hours = min_segment_hours
        self.min_stop_hours = min_stop_hours
    
    def classify_region(self, lon: float, lat: float) -> str:
        """根据经纬度判断区域"""
        # 优先匹配特定区域
        for region_id, region in self.PORT_REGIONS.items():
            if region_id == 'pacific':
                continue
            lon_min, lon_max = region['lon_range']
            lat_min, lat_max = region['lat_range']
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                return region['name']
        return '其他'
    
    def extract_segments(self, df: pd.DataFrame) -> Tuple[List[VoyageSegment], List[PortStop]]:
        """
        从船舶数据中提取航段和停靠信息
        
        Args:
            df: 单艘船的AIS数据，需包含 postime, lon, lat, sog 列
            
        Returns:
            (航段列表, 停靠列表)
        """
        if len(df) == 0:
            return [], []
        
        # 按时间排序
        df = df.sort_values('postime').copy()
        df['postime'] = pd.to_datetime(df['postime'])
        
        # 标记停靠状态
        df['is_stopped'] = df['sog'] < self.stop_speed_threshold
        
        # 分割连续的航行/停靠段
        df['segment_change'] = df['is_stopped'].ne(df['is_stopped'].shift()).cumsum()
        
        voyage_segments = []
        port_stops = []
        
        for segment_id, group in df.groupby('segment_change'):
            is_sailing = not group['is_stopped'].iloc[0]
            
            start_time = group['postime'].iloc[0]
            end_time = group['postime'].iloc[-1]
            duration_hours = (end_time - start_time).total_seconds() / 3600
            
            if is_sailing and duration_hours >= self.min_segment_hours:
                # 航行段
                segment = VoyageSegment(
                    segment_id=segment_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration_hours=duration_hours,
                    start_lon=group['lon'].iloc[0],
                    end_lon=group['lon'].iloc[-1],
                    start_lat=group['lat'].iloc[0],
                    end_lat=group['lat'].iloc[-1],
                    num_points=len(group),
                    avg_sog=group['sog'].mean(),
                    data=group.drop(columns=['is_stopped', 'segment_change'])
                )
                voyage_segments.append(segment)
                
            elif not is_sailing and duration_hours >= self.min_stop_hours:
                # 港口停靠
                mean_lon = group['lon'].mean()
                mean_lat = group['lat'].mean()
                stop = PortStop(
                    port_id=segment_id,
                    arrival_time=start_time,
                    departure_time=end_time,
                    duration_hours=duration_hours,
                    lon=mean_lon,
                    lat=mean_lat,
                    region=self.classify_region(mean_lon, mean_lat)
                )
                port_stops.append(stop)
        
        return voyage_segments, port_stops
    
    def find_longest_segment(self, segments: List[VoyageSegment]) -> Optional[VoyageSegment]:
        """找出最长的航段（通常是跨太平洋航段）"""
        if not segments:
            return None
        return max(segments, key=lambda s: s.duration_hours)
    
    def find_transpacific_segment(self, segments: List[VoyageSegment]) -> Optional[VoyageSegment]:
        """
        找出跨太平洋航段
        特征：起点在亚洲（经度>100或<-150），终点在美西（-130<经度<-115）
        或者反向
        """
        for segment in sorted(segments, key=lambda s: s.duration_hours, reverse=True):
            # 检查是否跨太平洋
            start_in_asia = segment.start_lon > 100 or segment.start_lon < -150
            end_in_us_west = -130 < segment.end_lon < -115
            
            start_in_us_west = -130 < segment.start_lon < -115
            end_in_asia = segment.end_lon > 100 or segment.end_lon < -150
            
            if (start_in_asia and end_in_us_west) or (start_in_us_west and end_in_asia):
                return segment
        
        # 如果没找到明确的跨太平洋航段，返回最长航段
        return self.find_longest_segment(segments)
    
    def get_arrival_time_at_port(self, df: pd.DataFrame, port_lon: float, port_lat: float,
                                  tolerance_deg: float = 0.5) -> Optional[pd.Timestamp]:
        """
        计算船舶到达港口附近的时间（不包括等泊时间）
        
        Args:
            df: 船舶轨迹数据
            port_lon, port_lat: 港口位置
            tolerance_deg: 位置容差（度）
            
        Returns:
            到达时间
        """
        df = df.sort_values('postime').copy()
        
        # 找到第一次进入港口区域且速度开始下降的点
        df['dist_to_port'] = np.sqrt((df['lon'] - port_lon)**2 + (df['lat'] - port_lat)**2)
        near_port = df[df['dist_to_port'] < tolerance_deg]
        
        if len(near_port) == 0:
            return None
        
        # 返回第一次到达港口附近的时间
        return pd.to_datetime(near_port['postime'].iloc[0])


class VoyageDataProcessor:
    """航程数据处理器：批量处理多个文件"""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = VoyageExtractor()
        
    def process_all_files(self, max_files: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        处理所有数据文件
        
        Returns:
            (航段数据DataFrame, 停靠数据DataFrame)
        """
        # Filter out Mac system files (._*)
        csv_files = sorted([f for f in self.data_dir.glob('*-ais.csv') 
                           if not f.name.startswith('._')])
        if max_files:
            csv_files = csv_files[:max_files]
        
        all_voyage_data = []
        all_stop_data = []
        
        for csv_file in tqdm(csv_files, desc='Processing files'):
            try:
                voyage_df, stop_df = self.process_single_file(csv_file)
                if voyage_df is not None and len(voyage_df) > 0:
                    all_voyage_data.append(voyage_df)
                if stop_df is not None and len(stop_df) > 0:
                    all_stop_data.append(stop_df)
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
        
        # 合并所有数据
        voyage_combined = pd.concat(all_voyage_data, ignore_index=True) if all_voyage_data else pd.DataFrame()
        stop_combined = pd.concat(all_stop_data, ignore_index=True) if all_stop_data else pd.DataFrame()
        
        return voyage_combined, stop_combined
    
    def process_single_file(self, csv_file: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """处理单个CSV文件"""
        # 只读取必要的列以节省内存
        usecols = ['mmsi', 'postime', 'lon', 'lat', 'sog', 'cog', 'hdg', 'draught', 'eta', 'status']
        try:
            df = pd.read_csv(csv_file, usecols=usecols, low_memory=False)
        except ValueError:
            # 如果某些列不存在，读取所有列
            df = pd.read_csv(csv_file, low_memory=False)
        
        # 获取MMSI（假设每个文件是单艘船）
        mmsi = df['mmsi'].iloc[0] if 'mmsi' in df.columns else csv_file.stem
        
        # 提取航段和停靠
        segments, stops = self.extractor.extract_segments(df)
        
        if not segments:
            return None, None
        
        # 找到最长航段（跨太平洋）
        main_segment = self.extractor.find_transpacific_segment(segments)
        
        if main_segment is None:
            return None, None
        
        # 准备航段数据
        voyage_df = main_segment.data.copy()
        voyage_df['mmsi'] = mmsi
        voyage_df['voyage_duration_hours'] = main_segment.duration_hours
        voyage_df['file_source'] = csv_file.name
        
        # 计算到达时间（去除等泊时间）
        # 到达时间 = 航段结束时间
        voyage_df['actual_arrival'] = main_segment.end_time
        
        # 为每个数据点计算剩余航行时间
        voyage_df['postime'] = pd.to_datetime(voyage_df['postime'])
        voyage_df['remaining_hours'] = (main_segment.end_time - voyage_df['postime']).dt.total_seconds() / 3600
        
        # 准备停靠数据
        stop_records = []
        for stop in stops:
            stop_records.append({
                'mmsi': mmsi,
                'port_id': stop.port_id,
                'arrival_time': stop.arrival_time,
                'departure_time': stop.departure_time,
                'duration_hours': stop.duration_hours,
                'lon': stop.lon,
                'lat': stop.lat,
                'region': stop.region,
                'file_source': csv_file.name
            })
        stop_df = pd.DataFrame(stop_records) if stop_records else None
        
        return voyage_df, stop_df
    
    def save_processed_data(self, voyage_df: pd.DataFrame, stop_df: pd.DataFrame):
        """保存处理后的数据"""
        if len(voyage_df) > 0:
            voyage_path = self.output_dir / 'processed_voyages.csv'
            voyage_df.to_csv(voyage_path, index=False)
            print(f"Saved voyage data to {voyage_path}: {len(voyage_df)} records")
        
        if len(stop_df) > 0:
            stop_path = self.output_dir / 'port_stops.csv'
            stop_df.to_csv(stop_path, index=False)
            print(f"Saved stop data to {stop_path}: {len(stop_df)} records")


def extract_training_features(voyage_df: pd.DataFrame) -> pd.DataFrame:
    """
    从航程数据中提取训练特征
    
    Features:
        - lat, lon: 位置
        - sog: 对地速度
        - cog: 对地航向
        - remaining_hours: 剩余航行时间（目标）
    """
    feature_cols = ['lat', 'lon', 'sog', 'cog', 'remaining_hours', 'mmsi', 'postime']
    
    # 只保留需要的列
    available_cols = [c for c in feature_cols if c in voyage_df.columns]
    result = voyage_df[available_cols].copy()
    
    # 清理数据
    result = result.dropna(subset=['lat', 'lon', 'sog', 'remaining_hours'])
    
    # 过滤异常值
    result = result[result['remaining_hours'] >= 0]
    result = result[result['remaining_hours'] <= 720]  # 最多30天
    result = result[result['sog'] >= 0]
    result = result[result['sog'] <= 30]  # 最大30节
    
    return result


if __name__ == '__main__':
    # 测试
    processor = VoyageDataProcessor('./data', './output/processed')
    voyage_df, stop_df = processor.process_all_files(max_files=3)
    
    print(f"\nVoyage data shape: {voyage_df.shape}")
    print(f"Stop data shape: {stop_df.shape}")
    
    if len(voyage_df) > 0:
        print("\nVoyage data sample:")
        print(voyage_df[['mmsi', 'lat', 'lon', 'sog', 'remaining_hours']].head())
    
    if len(stop_df) > 0:
        print("\nStop data sample:")
        print(stop_df.head())
        print("\nStop duration by region:")
        print(stop_df.groupby('region')['duration_hours'].agg(['mean', 'std', 'count']))
    
    # 保存
    processor.save_processed_data(voyage_df, stop_df)
