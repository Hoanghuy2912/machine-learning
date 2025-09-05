"""
Module phân tích biến động sử dụng đất/lớp phủ bề mặt đất theo thời gian
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
import yaml
from pathlib import Path

class LandUseChangeAnalyzer:
    """Class phân tích biến động sử dụng đất theo thời gian"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Khởi tạo change analyzer
        
        Args:
            config_path: Đường dẫn file cấu hình
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.land_cover_classes = self.config['land_cover_classes']
        
    def _load_config(self, config_path: str) -> Dict:
        """Đọc file cấu hình"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """Thiết lập logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def calculate_area_statistics(self, 
                                 classification_map: np.ndarray,
                                 pixel_size: float) -> pd.DataFrame:
        """
        Tính toán thống kê diện tích từng loại đất
        
        Args:
            classification_map: Bản đồ phân loại
            pixel_size: Kích thước pixel (m²)
            
        Returns:
            pd.DataFrame: Thống kê diện tích
        """
        unique_classes, counts = np.unique(classification_map, return_counts=True)
        
        # Tính diện tích (ha)
        areas_ha = (counts * pixel_size) / 10000  # Convert m² to ha
        
        # Tính tỷ lệ phần trăm
        total_area = np.sum(areas_ha)
        percentages = (areas_ha / total_area) * 100
        
        # Tạo DataFrame
        stats_data = []
        for class_id, area, percentage in zip(unique_classes, areas_ha, percentages):
            class_name = self._get_class_name(class_id)
            stats_data.append({
                'Class_ID': class_id,
                'Class_Name': class_name,
                'Area_ha': area,
                'Percentage': percentage
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        self.logger.info(f"Đã tính thống kê diện tích cho {len(unique_classes)} lớp")
        return stats_df
    
    def _get_class_name(self, class_id: int) -> str:
        """Lấy tên lớp từ ID"""
        for class_key, class_info in self.land_cover_classes.items():
            if class_info['id'] == class_id:
                return class_info['name']
        return f"Unknown_{class_id}"
    
    def create_change_matrix(self, 
                           map_t1: np.ndarray,
                           map_t2: np.ndarray) -> pd.DataFrame:
        """
        Tạo ma trận biến động giữa hai thời điểm
        
        Args:
            map_t1: Bản đồ thời điểm 1
            map_t2: Bản đồ thời điểm 2
            
        Returns:
            pd.DataFrame: Ma trận biến động
        """
        # Flatten arrays
        map_t1_flat = map_t1.flatten()
        map_t2_flat = map_t2.flatten()
        
        # Loại bỏ NoData
        valid_mask = (map_t1_flat > 0) & (map_t2_flat > 0)
        map_t1_valid = map_t1_flat[valid_mask]
        map_t2_valid = map_t2_flat[valid_mask]
        
        # Lấy unique classes
        all_classes = np.unique(np.concatenate([map_t1_valid, map_t2_valid]))
        
        # Tạo ma trận biến động
        change_matrix = np.zeros((len(all_classes), len(all_classes)), dtype=int)
        
        for i, class_from in enumerate(all_classes):
            for j, class_to in enumerate(all_classes):
                count = np.sum((map_t1_valid == class_from) & (map_t2_valid == class_to))
                change_matrix[i, j] = count
        
        # Tạo DataFrame với tên lớp
        class_names = [self._get_class_name(cls) for cls in all_classes]
        change_df = pd.DataFrame(
            change_matrix,
            index=class_names,
            columns=class_names
        )
        
        self.logger.info("Đã tạo ma trận biến động")
        return change_df
    
    def calculate_change_rates(self, 
                              area_stats_t1: pd.DataFrame,
                              area_stats_t2: pd.DataFrame,
                              years_between: int) -> pd.DataFrame:
        """
        Tính tốc độ thay đổi hàng năm
        
        Args:
            area_stats_t1: Thống kê diện tích thời điểm 1
            area_stats_t2: Thống kê diện tích thời điểm 2
            years_between: Số năm giữa hai thời điểm
            
        Returns:
            pd.DataFrame: Tốc độ thay đổi
        """
        # Merge DataFrames
        merged = pd.merge(
            area_stats_t1[['Class_Name', 'Area_ha']], 
            area_stats_t2[['Class_Name', 'Area_ha']], 
            on='Class_Name', 
            suffixes=('_t1', '_t2'),
            how='outer'
        ).fillna(0)
        
        # Tính thay đổi tuyệt đối và tương đối
        merged['Change_ha'] = merged['Area_ha_t2'] - merged['Area_ha_t1']
        merged['Change_percent'] = ((merged['Area_ha_t2'] - merged['Area_ha_t1']) / 
                                   merged['Area_ha_t1']) * 100
        merged['Change_percent'] = merged['Change_percent'].replace([np.inf, -np.inf], 0)
        
        # Tốc độ thay đổi hàng năm
        merged['Annual_change_rate'] = merged['Change_percent'] / years_between
        
        self.logger.info("Đã tính tốc độ thay đổi")
        return merged
    
    def identify_hotspots(self, 
                         change_map: np.ndarray,
                         window_size: int = 5,
                         threshold: float = 0.5) -> np.ndarray:
        """
        Xác định các điểm nóng biến động
        
        Args:
            change_map: Bản đồ biến động (binary: 0=no change, 1=change)
            window_size: Kích thước cửa sổ phân tích
            threshold: Ngưỡng tỷ lệ thay đổi
            
        Returns:
            np.ndarray: Bản đồ điểm nóng
        """
        from scipy import ndimage
        
        # Tính tỷ lệ thay đổi trong từng cửa sổ
        kernel = np.ones((window_size, window_size))
        change_density = ndimage.convolve(
            change_map.astype(float), kernel, mode='constant'
        ) / (window_size * window_size)
        
        # Xác định hotspots
        hotspots = (change_density > threshold).astype(int)
        
        self.logger.info(f"Đã xác định {np.sum(hotspots)} điểm nóng biến động")
        return hotspots
    
    def analyze_spatial_patterns(self, 
                                classification_map: np.ndarray) -> Dict:
        """
        Phân tích các mẫu không gian của sử dụng đất
        
        Args:
            classification_map: Bản đồ phân loại
            
        Returns:
            Dict: Thống kê mẫu không gian
        """
        from scipy.ndimage import label
        from skimage.measure import regionprops
        
        results = {}
        
        for class_key, class_info in self.land_cover_classes.items():
            class_id = class_info['id']
            class_name = class_info['name']
            
            # Tạo binary mask cho lớp này
            class_mask = (classification_map == class_id)
            
            if np.sum(class_mask) == 0:
                continue
            
            # Label connected components
            labeled_array, num_features = label(class_mask)
            
            # Tính các thuộc tính không gian
            props = regionprops(labeled_array)
            
            if props:
                areas = [prop.area for prop in props]
                perimeters = [prop.perimeter for prop in props]
                
                # Shape index (tỷ lệ chu vi/diện tích)
                shape_indices = [p / (2 * np.sqrt(np.pi * a)) for a, p in zip(areas, perimeters)]
                
                results[class_name] = {
                    'num_patches': num_features,
                    'total_area': np.sum(areas),
                    'mean_patch_size': np.mean(areas),
                    'largest_patch': np.max(areas),
                    'mean_shape_index': np.mean(shape_indices),
                    'patch_density': num_features / np.sum(class_mask)
                }
        
        self.logger.info("Đã phân tích mẫu không gian")
        return results
    
    def calculate_landscape_metrics(self, 
                                  classification_map: np.ndarray) -> Dict:
        """
        Tính toán các chỉ số cảnh quan
        
        Args:
            classification_map: Bản đồ phân loại
            
        Returns:
            Dict: Các chỉ số cảnh quan
        """
        from scipy.ndimage import label
        
        metrics = {}
        
        # Tổng số lớp
        unique_classes = np.unique(classification_map[classification_map > 0])
        metrics['num_classes'] = len(unique_classes)
        
        # Shannon Diversity Index
        total_pixels = np.sum(classification_map > 0)
        shannon_diversity = 0
        
        for class_id in unique_classes:
            proportion = np.sum(classification_map == class_id) / total_pixels
            if proportion > 0:
                shannon_diversity -= proportion * np.log(proportion)
        
        metrics['shannon_diversity'] = shannon_diversity
        
        # Simpson's Diversity Index
        simpson_diversity = 0
        for class_id in unique_classes:
            proportion = np.sum(classification_map == class_id) / total_pixels
            simpson_diversity += proportion ** 2
        
        metrics['simpson_diversity'] = 1 - simpson_diversity
        
        # Evenness
        max_diversity = np.log(len(unique_classes))
        metrics['evenness'] = shannon_diversity / max_diversity if max_diversity > 0 else 0
        
        # Fragmentation metrics
        total_patches = 0
        total_edge = 0
        
        for class_id in unique_classes:
            class_mask = (classification_map == class_id)
            labeled_array, num_patches = label(class_mask)
            total_patches += num_patches
            
            # Tính edge density (simplified)
            edges = np.sum(np.diff(class_mask.astype(int), axis=0) != 0) + \
                   np.sum(np.diff(class_mask.astype(int), axis=1) != 0)
            total_edge += edges
        
        metrics['patch_density'] = total_patches / total_pixels
        metrics['edge_density'] = total_edge / total_pixels
        
        self.logger.info("Đã tính toán các chỉ số cảnh quan")
        return metrics
    
    def create_change_trajectories(self, 
                                 maps_time_series: List[np.ndarray],
                                 years: List[int]) -> Dict:
        """
        Tạo quỹ đạo biến động qua nhiều thời điểm
        
        Args:
            maps_time_series: Danh sách bản đồ phân loại theo thời gian
            years: Danh sách năm tương ứng
            
        Returns:
            Dict: Quỹ đạo biến động
        """
        if len(maps_time_series) != len(years):
            raise ValueError("Số lượng bản đồ và năm phải bằng nhau")
        
        trajectories = {}
        
        # Flatten tất cả bản đồ
        flattened_maps = [m.flatten() for m in maps_time_series]
        
        # Tạo trajectory code cho mỗi pixel
        num_pixels = len(flattened_maps[0])
        trajectory_codes = []
        
        for pixel_idx in range(num_pixels):
            trajectory = tuple(m[pixel_idx] for m in flattened_maps)
            trajectory_codes.append(trajectory)
        
        # Đếm từng loại trajectory
        unique_trajectories, counts = np.unique(trajectory_codes, return_counts=True, axis=0)
        
        # Chuyển đổi thành dictionary
        for trajectory, count in zip(unique_trajectories, counts):
            trajectory_str = "->".join([self._get_class_name(cls) for cls in trajectory])
            trajectories[trajectory_str] = count
        
        # Sắp xếp theo số lượng giảm dần
        trajectories = dict(sorted(trajectories.items(), key=lambda x: x[1], reverse=True))
        
        self.logger.info(f"Đã tạo {len(trajectories)} loại quỹ đạo biến động")
        return trajectories
    
    def export_change_statistics(self, 
                                analysis_results: Dict,
                                output_path: str) -> str:
        """
        Xuất thống kê biến động ra file
        
        Args:
            analysis_results: Kết quả phân tích
            output_path: Đường dẫn file xuất
            
        Returns:
            str: Đường dẫn file đã xuất
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Xuất từng bảng thống kê
            if 'area_statistics' in analysis_results:
                for year, stats in analysis_results['area_statistics'].items():
                    stats.to_excel(writer, sheet_name=f'Area_Stats_{year}', index=False)
            
            if 'change_matrices' in analysis_results:
                for period, matrix in analysis_results['change_matrices'].items():
                    matrix.to_excel(writer, sheet_name=f'Change_Matrix_{period}')
            
            if 'change_rates' in analysis_results:
                for period, rates in analysis_results['change_rates'].items():
                    rates.to_excel(writer, sheet_name=f'Change_Rates_{period}', index=False)
            
            if 'landscape_metrics' in analysis_results:
                metrics_df = pd.DataFrame(analysis_results['landscape_metrics']).T
                metrics_df.to_excel(writer, sheet_name='Landscape_Metrics')
        
        self.logger.info(f"Đã xuất thống kê biến động tại: {output_path}")
        return output_path
    
    def generate_change_report(self, 
                             analysis_results: Dict,
                             output_path: str) -> str:
        """
        Tạo báo cáo tổng hợp biến động
        
        Args:
            analysis_results: Kết quả phân tích
            output_path: Đường dẫn file báo cáo
            
        Returns:
            str: Đường dẫn file báo cáo
        """
        report_content = []
        
        # Header
        report_content.append("# BÁO CÁO PHÂN TÍCH BIẾN ĐỘNG SỬ DỤNG ĐẤT TỈNH ĐỒNG THÁP\n")
        report_content.append(f"Ngày tạo: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}\n\n")
        
        # Tóm tắt nghiên cứu
        report_content.append("## 1. TỔNG QUAN\n")
        report_content.append("Phân tích biến động sử dụng đất/lớp phủ bề mặt đất tỉnh Đồng Tháp ")
        report_content.append("giai đoạn 1990-2025 sử dụng công nghệ viễn thám và học máy.\n\n")
        
        # Thống kê diện tích
        if 'area_statistics' in analysis_results:
            report_content.append("## 2. THỐNG KÊ DIỆN TÍCH THEO THỜI GIAN\n")
            for year, stats in analysis_results['area_statistics'].items():
                report_content.append(f"### Năm {year}\n")
                report_content.append(stats.to_string(index=False))
                report_content.append("\n\n")
        
        # Tốc độ thay đổi
        if 'change_rates' in analysis_results:
            report_content.append("## 3. TỐC ĐỘ THAY ĐỔI\n")
            for period, rates in analysis_results['change_rates'].items():
                report_content.append(f"### Giai đoạn {period}\n")
                significant_changes = rates[abs(rates['Annual_change_rate']) > 1]
                if not significant_changes.empty:
                    report_content.append("Các loại đất có thay đổi đáng kể (>1%/năm):\n")
                    report_content.append(significant_changes.to_string(index=False))
                report_content.append("\n\n")
        
        # Xu hướng và nhận xét
        report_content.append("## 4. NHẬN XÉT VÀ XU HƯỚNG\n")
        report_content.append("- Phân tích xu hướng biến động chính\n")
        report_content.append("- Các yếu tố tác động\n")
        report_content.append("- Đề xuất chính sách quản lý\n\n")
        
        # Kết luận
        report_content.append("## 5. KẾT LUẬN VÀ KHUYẾN NGHỊ\n")
        report_content.append("- Kết luận chính từ phân tích\n")
        report_content.append("- Khuyến nghị cho quy hoạch sử dụng đất\n")
        report_content.append("- Định hướng phát triển bền vững\n")
        
        # Lưu báo cáo
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(report_content))
        
        self.logger.info(f"Đã tạo báo cáo tại: {output_path}")
        return output_path
