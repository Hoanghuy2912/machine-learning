"""
Module tiền xử lý dữ liệu ảnh viễn thám cho phân tích biến động sử dụng đất
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional, Union
import logging
import yaml
from pathlib import Path
import cv2

class SatelliteDataPreprocessor:
    """Class xử lý và chuẩn bị dữ liệu ảnh viễn thám"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Khởi tạo preprocessor
        
        Args:
            config_path: Đường dẫn file cấu hình
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.scaler = None
        
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
    
    def load_raster(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Đọc file raster
        
        Args:
            file_path: Đường dẫn file raster
            
        Returns:
            Tuple[np.ndarray, Dict]: Dữ liệu ảnh và metadata
        """
        with rasterio.open(file_path) as src:
            data = src.read()
            metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': src.dtype,
                'nodata': src.nodata
            }
            
        self.logger.info(f"Đã đọc raster {file_path}: {data.shape}")
        return data, metadata
    
    def clip_raster_to_boundary(self, 
                               raster_path: str,
                               boundary_path: str,
                               output_path: str) -> str:
        """
        Cắt raster theo ranh giới hành chính
        
        Args:
            raster_path: Đường dẫn raster gốc
            boundary_path: Đường dẫn shapefile ranh giới
            output_path: Đường dẫn file đầu ra
            
        Returns:
            str: Đường dẫn file đã cắt
        """
        # Đọc ranh giới
        boundary = gpd.read_file(boundary_path)
        
        # Đọc raster
        with rasterio.open(raster_path) as src:
            # Chuyển đổi geometry sang cùng CRS với raster
            boundary_reproj = boundary.to_crs(src.crs)
            
            # Cắt raster
            out_image, out_transform = mask(
                src, boundary_reproj.geometry, crop=True
            )
            out_meta = src.meta.copy()
            
            # Cập nhật metadata
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Lưu file
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
        
        self.logger.info(f"Đã cắt raster và lưu tại: {output_path}")
        return output_path
    
    def reproject_raster(self,
                        input_path: str,
                        output_path: str,
                        target_crs: str = "EPSG:3405") -> str:
        """
        Chuyển đổi hệ tọa độ raster
        
        Args:
            input_path: Đường dẫn raster gốc
            output_path: Đường dẫn file đầu ra
            target_crs: Hệ tọa độ đích
            
        Returns:
            str: Đường dẫn file đã chuyển đổi
        """
        with rasterio.open(input_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest
                    )
        
        self.logger.info(f"Đã chuyển đổi CRS và lưu tại: {output_path}")
        return output_path
    
    def resample_raster(self,
                       input_path: str,
                       output_path: str,
                       target_resolution: float) -> str:
        """
        Thay đổi độ phân giải raster
        
        Args:
            input_path: Đường dẫn raster gốc
            output_path: Đường dẫn file đầu ra
            target_resolution: Độ phân giải đích (meters)
            
        Returns:
            str: Đường dẫn file đã thay đổi độ phân giải
        """
        with rasterio.open(input_path) as src:
            # Tính toán kích thước mới
            scale_factor = src.res[0] / target_resolution
            new_width = int(src.width * scale_factor)
            new_height = int(src.height * scale_factor)
            
            # Tạo transform mới
            new_transform = src.transform * src.transform.scale(
                (src.width / new_width),
                (src.height / new_height)
            )
            
            kwargs = src.meta.copy()
            kwargs.update({
                'transform': new_transform,
                'width': new_width,
                'height': new_height
            })
            
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=new_transform,
                        dst_crs=src.crs,
                        resampling=Resampling.bilinear
                    )
        
        self.logger.info(f"Đã resample và lưu tại: {output_path}")
        return output_path
    
    def calculate_spectral_indices(self, 
                                  image_data: np.ndarray,
                                  band_mapping: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        Tính toán các chỉ số phổ
        
        Args:
            image_data: Dữ liệu ảnh (bands, height, width)
            band_mapping: Mapping tên band với index
            
        Returns:
            Dict[str, np.ndarray]: Dictionary các chỉ số phổ
        """
        indices = {}
        
        # NDVI = (NIR - RED) / (NIR + RED)
        if 'NIR' in band_mapping and 'RED' in band_mapping:
            nir = image_data[band_mapping['NIR']].astype(np.float32)
            red = image_data[band_mapping['RED']].astype(np.float32)
            indices['NDVI'] = np.divide(
                (nir - red),
                (nir + red),
                out=np.zeros_like(nir),
                where=(nir + red) != 0
            )
        
        # MNDWI = (GREEN - SWIR1) / (GREEN + SWIR1)
        if 'GREEN' in band_mapping and 'SWIR1' in band_mapping:
            green = image_data[band_mapping['GREEN']].astype(np.float32)
            swir1 = image_data[band_mapping['SWIR1']].astype(np.float32)
            indices['MNDWI'] = np.divide(
                (green - swir1),
                (green + swir1),
                out=np.zeros_like(green),
                where=(green + swir1) != 0
            )
        
        # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
        if 'SWIR1' in band_mapping and 'NIR' in band_mapping:
            swir1 = image_data[band_mapping['SWIR1']].astype(np.float32)
            nir = image_data[band_mapping['NIR']].astype(np.float32)
            indices['NDBI'] = np.divide(
                (swir1 - nir),
                (swir1 + nir),
                out=np.zeros_like(swir1),
                where=(swir1 + nir) != 0
            )
        
        # EVI = 2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))
        if all(band in band_mapping for band in ['NIR', 'RED', 'BLUE']):
            nir = image_data[band_mapping['NIR']].astype(np.float32)
            red = image_data[band_mapping['RED']].astype(np.float32)
            blue = image_data[band_mapping['BLUE']].astype(np.float32)
            
            denominator = nir + 6*red - 7.5*blue + 1
            indices['EVI'] = np.divide(
                2.5 * (nir - red),
                denominator,
                out=np.zeros_like(nir),
                where=denominator != 0
            )
        
        self.logger.info(f"Đã tính toán {len(indices)} chỉ số phổ")
        return indices
    
    def normalize_data(self, 
                      data: np.ndarray,
                      method: str = 'minmax',
                      fit_scaler: bool = True) -> np.ndarray:
        """
        Chuẩn hóa dữ liệu
        
        Args:
            data: Dữ liệu đầu vào (samples, features)
            method: Phương pháp chuẩn hóa ('minmax', 'standard')
            fit_scaler: Có fit scaler hay không
            
        Returns:
            np.ndarray: Dữ liệu đã chuẩn hóa
        """
        if method == 'minmax':
            if self.scaler is None or fit_scaler:
                self.scaler = MinMaxScaler()
                normalized_data = self.scaler.fit_transform(data)
            else:
                normalized_data = self.scaler.transform(data)
                
        elif method == 'standard':
            if self.scaler is None or fit_scaler:
                self.scaler = StandardScaler()
                normalized_data = self.scaler.fit_transform(data)
            else:
                normalized_data = self.scaler.transform(data)
        else:
            raise ValueError(f"Phương pháp không hỗ trợ: {method}")
        
        self.logger.info(f"Đã chuẩn hóa dữ liệu bằng phương pháp {method}")
        return normalized_data
    
    def create_image_patches(self,
                           image_data: np.ndarray,
                           patch_size: int = 64,
                           overlap: float = 0.1) -> List[np.ndarray]:
        """
        Tạo các patch nhỏ từ ảnh lớn cho deep learning
        
        Args:
            image_data: Dữ liệu ảnh (height, width, bands)
            patch_size: Kích thước patch
            overlap: Tỷ lệ overlap giữa các patch
            
        Returns:
            List[np.ndarray]: Danh sách các patch
        """
        height, width, bands = image_data.shape
        step = int(patch_size * (1 - overlap))
        patches = []
        
        for y in range(0, height - patch_size + 1, step):
            for x in range(0, width - patch_size + 1, step):
                patch = image_data[y:y+patch_size, x:x+patch_size, :]
                patches.append(patch)
        
        self.logger.info(f"Đã tạo {len(patches)} patches từ ảnh {height}x{width}")
        return patches
    
    def apply_noise_reduction(self,
                            image_data: np.ndarray,
                            method: str = 'gaussian') -> np.ndarray:
        """
        Áp dụng bộ lọc giảm nhiễu
        
        Args:
            image_data: Dữ liệu ảnh
            method: Phương pháp lọc ('gaussian', 'median', 'bilateral')
            
        Returns:
            np.ndarray: Ảnh đã lọc nhiễu
        """
        if len(image_data.shape) == 3:  # Multi-band image
            filtered_data = np.zeros_like(image_data)
            for i in range(image_data.shape[0]):
                band = image_data[i]
                if method == 'gaussian':
                    filtered_data[i] = cv2.GaussianBlur(band, (5, 5), 0)
                elif method == 'median':
                    filtered_data[i] = cv2.medianBlur(band.astype(np.uint8), 5)
                elif method == 'bilateral':
                    filtered_data[i] = cv2.bilateralFilter(
                        band.astype(np.uint8), 9, 75, 75
                    )
        else:  # Single band
            if method == 'gaussian':
                filtered_data = cv2.GaussianBlur(image_data, (5, 5), 0)
            elif method == 'median':
                filtered_data = cv2.medianBlur(image_data.astype(np.uint8), 5)
            elif method == 'bilateral':
                filtered_data = cv2.bilateralFilter(
                    image_data.astype(np.uint8), 9, 75, 75
                )
        
        self.logger.info(f"Đã áp dụng bộ lọc {method}")
        return filtered_data
    
    def extract_training_data(self,
                            image_path: str,
                            reference_path: str,
                            sample_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trích xuất dữ liệu training từ ảnh và dữ liệu tham chiếu
        
        Args:
            image_path: Đường dẫn ảnh vệ tinh
            reference_path: Đường dẫn ảnh tham chiếu (ground truth)
            sample_size: Số lượng mẫu cho mỗi lớp
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features và labels
        """
        # Đọc ảnh vệ tinh
        image_data, _ = self.load_raster(image_path)
        
        # Đọc dữ liệu tham chiếu
        reference_data, _ = self.load_raster(reference_path)
        reference_data = reference_data[0]  # Chỉ lấy band đầu tiên
        
        # Reshape dữ liệu
        height, width = image_data.shape[1], image_data.shape[2]
        features = image_data.reshape(image_data.shape[0], -1).T
        labels = reference_data.reshape(-1)
        
        # Loại bỏ pixels có giá trị NoData
        valid_mask = labels > 0
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        # Lấy mẫu cân bằng từ mỗi lớp
        unique_classes = np.unique(labels)
        sampled_features = []
        sampled_labels = []
        
        for class_id in unique_classes:
            class_mask = labels == class_id
            class_features = features[class_mask]
            class_labels = labels[class_mask]
            
            if len(class_features) > sample_size:
                indices = np.random.choice(
                    len(class_features), sample_size, replace=False
                )
                sampled_features.append(class_features[indices])
                sampled_labels.append(class_labels[indices])
            else:
                sampled_features.append(class_features)
                sampled_labels.append(class_labels)
        
        final_features = np.vstack(sampled_features)
        final_labels = np.hstack(sampled_labels)
        
        self.logger.info(f"Đã trích xuất {len(final_features)} mẫu training "
                        f"từ {len(unique_classes)} lớp")
        
        return final_features, final_labels
    
    def save_processed_data(self,
                          data: np.ndarray,
                          output_path: str,
                          metadata: Dict) -> str:
        """
        Lưu dữ liệu đã xử lý
        
        Args:
            data: Dữ liệu ảnh
            output_path: Đường dẫn file đầu ra
            metadata: Metadata của ảnh
            
        Returns:
            str: Đường dẫn file đã lưu
        """
        with rasterio.open(output_path, 'w', **metadata) as dst:
            if len(data.shape) == 3:
                dst.write(data)
            else:
                dst.write(data, 1)
        
        self.logger.info(f"Đã lưu dữ liệu tại: {output_path}")
        return output_path
