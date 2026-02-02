#!/usr/bin/env python
"""
数据预处理脚本

功能：
1. 从原始AIS数据中提取最长航段（跨太平洋航段）
2. 去除港口等泊时间，只保留纯航行数据
3. 提取港口停靠信息用于训练停靠时间模型

用法：
    python preprocess_data.py --data_dir ./data --output_dir ./output/processed --max_files 5
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Optional
from dataclasses import dataclass
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
    data: pd.DataFrame


@dataclass
class PortStop:
    """港口停靠信息"""
    port_id: int
    arrival_time: pd.Timestamp
    departure_time: pd.Timestamp
    duration_hours: float
    lon: float
    lat: float
    region: str


class VoyageExtractor:
    """从AIS数据中提取航程信息"""
    
    # 速度阈值：低于此值视为停靠（节）- 降低阈值更严格判定停靠
    STOP_SPEED_THRESHOLD = 0.3
    
    # 最小航段时长（小时）
    MIN_SEGMENT_HOURS = 6
    
    # 最小停靠时长（小时）- 增加到2小时避免误判
    MIN_STOP_HOURS = 2
    
    # 最小数据密度（点/天）
    MIN_POINTS_PER_DAY = 10
    
    def __init__(self, 
                 stop_speed_threshold: float = 0.3,
                 min_segment_hours: float = 6,
                 min_stop_hours: float = 2,
                 min_points_per_day: float = 10):
        self.stop_speed_threshold = stop_speed_threshold
        self.min_segment_hours = min_segment_hours
        self.min_stop_hours = min_stop_hours
        self.min_points_per_day = min_points_per_day
    
    def classify_region(self, lon: float, lat: float) -> str:
        """根据经纬度判断区域"""
        # 中国东部
        if 100 <= lon <= 135 and 20 <= lat <= 45:
            return '中国东部'
        # 新加坡
        if 103 <= lon <= 105 and 0 <= lat <= 3:
            return '新加坡'
        # 中东/印度
        if 55 <= lon <= 80 and 10 <= lat <= 30:
            return '中东/印度'
        # 红海/苏伊士
        if 30 <= lon <= 55 and 25 <= lat <= 35:
            return '红海/苏伊士'
        # 美国西海岸
        if -130 <= lon <= -115 and 30 <= lat <= 50:
            return '美国西海岸'
        return '其他'
    
    def extract_segments(self, df: pd.DataFrame) -> Tuple[List[VoyageSegment], List[PortStop]]:
        """从船舶数据中提取航段和停靠信息"""
        if len(df) < 10:
            return [], []
        
        df = df.sort_values('postime').copy()
        df['postime'] = pd.to_datetime(df['postime'])
        
        # 检查数据密度
        time_span_days = (df['postime'].max() - df['postime'].min()).total_seconds() / 86400
        if time_span_days > 0:
            points_per_day = len(df) / time_span_days
            if points_per_day < self.min_points_per_day:
                # 数据太稀疏，跳过
                return [], []
        
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
        """找出最长的航段"""
        if not segments:
            return None
        return max(segments, key=lambda s: s.duration_hours)
    
    def find_transpacific_segment(self, segments: List[VoyageSegment]) -> Optional[VoyageSegment]:
        """找出跨太平洋航段"""
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


class DataProcessor:
    """批量处理数据"""
    
    # 只读取需要的列
    REQUIRED_COLS = ['mmsi', 'postime', 'eta', 'lon', 'lat', 'sog', 'cog']
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = VoyageExtractor()
    
    def get_ais_files(self) -> List[Path]:
        """获取所有AIS数据文件"""
        files = []
        for f in self.data_dir.glob('*-ais.csv'):
            # 排除Mac系统文件
            if not f.name.startswith('._'):
                files.append(f)
        return sorted(files)
    
    def process_single_file(self, csv_file: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """处理单个CSV文件"""
        try:
            # 只读取需要的列
            df = pd.read_csv(csv_file, usecols=self.REQUIRED_COLS, low_memory=False)
        except ValueError:
            # 如果某些列不存在，读取全部再筛选
            df = pd.read_csv(csv_file, low_memory=False)
            available_cols = [c for c in self.REQUIRED_COLS if c in df.columns]
            df = df[available_cols]
        
        # 获取MMSI
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
        
        # 计算剩余航行时间
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
    
    def process_all(self, max_files: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """处理所有文件"""
        csv_files = self.get_ais_files()
        if max_files:
            csv_files = csv_files[:max_files]
        
        print(f"Found {len(csv_files)} AIS files to process")
        
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
                print(f"\nError processing {csv_file.name}: {e}")
                continue
        
        voyage_combined = pd.concat(all_voyage_data, ignore_index=True) if all_voyage_data else pd.DataFrame()
        stop_combined = pd.concat(all_stop_data, ignore_index=True) if all_stop_data else pd.DataFrame()
        
        return voyage_combined, stop_combined
    
    def save(self, voyage_df: pd.DataFrame, stop_df: pd.DataFrame):
        """保存处理后的数据"""
        if len(voyage_df) > 0:
            voyage_path = self.output_dir / 'processed_voyages.csv'
            voyage_df.to_csv(voyage_path, index=False)
            print(f"Saved voyage data: {len(voyage_df)} records -> {voyage_path}")
        
        if len(stop_df) > 0:
            # 过滤异常停靠时间（超过7天=168小时的可能是数据问题）
            stop_df = stop_df[(stop_df['duration_hours'] >= 1) & (stop_df['duration_hours'] <= 168)]
            stop_path = self.output_dir / 'port_stops.csv'
            stop_df.to_csv(stop_path, index=False)
            print(f"Saved stop data: {len(stop_df)} records -> {stop_path}")


def print_summary(voyage_df: pd.DataFrame, stop_df: pd.DataFrame):
    """打印数据摘要"""
    print("\n" + "="*50)
    print("数据摘要")
    print("="*50)
    
    if len(voyage_df) > 0:
        print(f"\n【航程数据】")
        print(f"  总数据点: {len(voyage_df):,}")
        print(f"  船舶数量: {voyage_df['mmsi'].nunique()}")
        print(f"  剩余时间范围: {voyage_df['remaining_hours'].min():.1f} ~ {voyage_df['remaining_hours'].max():.1f} 小时")
        print(f"  速度范围: {voyage_df['sog'].min():.1f} ~ {voyage_df['sog'].max():.1f} 节")
        
        # 按船舶统计
        print(f"\n  各船航程时长:")
        for mmsi, group in voyage_df.groupby('mmsi'):
            duration = group['voyage_duration_hours'].iloc[0]
            print(f"    {mmsi}: {duration:.1f} 小时 ({duration/24:.1f} 天), {len(group):,} 点")
    
    if len(stop_df) > 0:
        print(f"\n【港口停靠数据】")
        print(f"  总停靠次数: {len(stop_df)}")
        print(f"  停靠时长: {stop_df['duration_hours'].min():.1f} ~ {stop_df['duration_hours'].max():.1f} 小时")
        
        print(f"\n  按区域统计:")
        region_stats = stop_df.groupby('region')['duration_hours'].agg(['mean', 'count'])
        for region, row in region_stats.iterrows():
            print(f"    {region}: 平均 {row['mean']:.1f} 小时, {int(row['count'])} 次")


def filter_data_quality(voyage_df: pd.DataFrame, stop_df: pd.DataFrame, 
                        min_points: int = 50, 
                        min_points_per_day: float = 10.0,
                        max_stop_hours: float = 168.0,
                        min_stop_hours: float = 4.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """数据质量过滤"""
    print("\n数据质量过滤...")
    
    original_voyages = len(voyage_df)
    original_ships = voyage_df['mmsi'].nunique() if len(voyage_df) > 0 else 0
    
    # 过滤航程数据
    valid_ships = []
    if len(voyage_df) > 0:
        # 计算每艘船的数据密度
        ship_stats = []
        for mmsi, group in voyage_df.groupby('mmsi'):
            n_points = len(group)
            duration_hours = group['voyage_duration_hours'].iloc[0]
            duration_days = duration_hours / 24
            points_per_day = n_points / duration_days if duration_days > 0 else 0
            ship_stats.append({
                'mmsi': mmsi,
                'n_points': n_points,
                'duration_days': duration_days,
                'points_per_day': points_per_day
            })
        
        stats_df = pd.DataFrame(ship_stats)
        
        # 过滤条件：数据点>=50 且 每天至少10个点
        valid_ships = stats_df[
            (stats_df['n_points'] >= min_points) & 
            (stats_df['points_per_day'] >= min_points_per_day)
        ]['mmsi'].tolist()
        
        # 打印被过滤掉的船
        filtered_ships = stats_df[~stats_df['mmsi'].isin(valid_ships)]
        if len(filtered_ships) > 0:
            print(f"  过滤掉 {len(filtered_ships)} 艘数据稀疏的船:")
            for _, row in filtered_ships.iterrows():
                print(f"    {row['mmsi']}: {row['n_points']:.0f}点, {row['duration_days']:.1f}天, {row['points_per_day']:.1f}点/天")
        
        voyage_df = voyage_df[voyage_df['mmsi'].isin(valid_ships)].copy()
    
    # 过滤停靠数据
    if len(stop_df) > 0:
        original_stops = len(stop_df)
        # 1. 只保留有效船舶的停靠数据
        stop_df = stop_df[stop_df['mmsi'].isin(valid_ships)].copy()
        # 2. 过滤异常停靠时间（太短或太长）
        stop_df = stop_df[(stop_df['duration_hours'] >= min_stop_hours) & (stop_df['duration_hours'] <= max_stop_hours)].copy()
        print(f"  停靠数据: {original_stops} -> {len(stop_df)} (过滤无效船舶和异常停靠时间)")
    
    print(f"  航程数据: {original_voyages} -> {len(voyage_df)} ({original_ships} -> {voyage_df['mmsi'].nunique()} 艘船)")
    
    return voyage_df, stop_df


def main():
    parser = argparse.ArgumentParser(description='AIS数据预处理')
    parser.add_argument('--data_dir', type=str, default='./data', help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='./output/processed', help='输出目录')
    parser.add_argument('--max_files', type=int, default=None, help='最多处理文件数（用于测试）')
    parser.add_argument('--min_points', type=int, default=50, help='每艘船最少数据点数')
    parser.add_argument('--min_points_per_day', type=float, default=10.0, help='每天最少数据点数')
    
    args = parser.parse_args()
    
    print("="*50)
    print("AIS数据预处理")
    print("="*50)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    
    processor = DataProcessor(args.data_dir, args.output_dir)
    voyage_df, stop_df = processor.process_all(max_files=args.max_files)
    
    if len(voyage_df) == 0:
        print("\n警告: 未提取到有效航程数据!")
        return
    
    # 数据质量过滤
    voyage_df, stop_df = filter_data_quality(
        voyage_df, stop_df,
        min_points=args.min_points,
        min_points_per_day=args.min_points_per_day
    )
    
    if len(voyage_df) == 0:
        print("\n警告: 过滤后无有效数据!")
        return
    
    processor.save(voyage_df, stop_df)
    print_summary(voyage_df, stop_df)
    
    print("\n预处理完成!")


if __name__ == '__main__':
    main()
