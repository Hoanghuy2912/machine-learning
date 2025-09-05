"""
Google Earth Engine Client cho dự án phân tích biến động sử dụng đất tỉnh Đồng Tháp
"""

import ee
import os
import logging
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path

class GEEClient:
    """Client để kết nối và xử lý dữ liệu Google Earth Engine"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Khởi tạo GEE Client
        
        Args:
            config_path: Đường dẫn đến file cấu hình
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.study_area = None
        self._initialize_gee()
        
    def _load_config(self, config_path: str) -> Dict:
        """Đọc file cấu hình"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")
    
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
    
    def _initialize_gee(self):
        """Khởi tạo và xác thực Google Earth Engine"""
        try:
            # Thử xác thực với service account (nếu có)
            if os.path.exists('service_account_key.json'):
                credentials = ee.ServiceAccountCredentials(
                    email=None,  # Sẽ được đọc từ file JSON
                    key_file='service_account_key.json'
                )
                ee.Initialize(credentials)
                self.logger.info("Đã xác thực GEE bằng service account")
            else:
                # Xác thực thông thường với fallback project
                try:
                    ee.Initialize()
                    self.logger.info("Đã xác thực GEE bằng tài khoản cá nhân")
                except ee.EEException as ee_error:
                    if "no project found" in str(ee_error):
                        # Thử với project mặc định
                        try:
                            ee.Initialize(project='ee-demo')  # Project demo
                            self.logger.info("Đã xác thực GEE với project demo")
                        except:
                            self.logger.error("Cần xác thực GEE và thiết lập project")
                            self.logger.info("Chạy: earthengine authenticate")
                            self.logger.info("Sau đó: earthengine set_project YOUR_PROJECT_ID")
                            raise
                    else:
                        raise
                
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo Google Earth Engine: {str(e)}")
            self.logger.info("Vui lòng chạy 'earthengine authenticate' để xác thực")
            self.logger.info("Và thiết lập project: earthengine set_project YOUR_PROJECT_ID")
            raise
    
    def set_study_area(self, boundary_path: Optional[str] = None) -> ee.Geometry:
        """
        Thiết lập khu vực nghiên cứu
        
        Args:
            boundary_path: Đường dẫn đến file shapefile ranh giới
            
        Returns:
            ee.Geometry: Vùng nghiên cứu
        """
        if boundary_path and os.path.exists(boundary_path):
            # Đọc từ shapefile (cần convert sang GEE format trước)
            self.logger.info(f"Đọc ranh giới từ file: {boundary_path}")
            # TODO: Implement shapefile reading
        else:
            # Sử dụng tọa độ từ config
            bounds = self.config['study_area']['bounds']
            self.study_area = ee.Geometry.Rectangle([
                bounds['west'], bounds['south'],
                bounds['east'], bounds['north']
            ])
            self.logger.info("Sử dụng ranh giới từ cấu hình")
            
        return self.study_area
    
    def get_landsat_collection(self, 
                              start_date: str, 
                              end_date: str,
                              cloud_threshold: int = 20) -> ee.ImageCollection:
        """
        Lấy dữ liệu Landsat cho khoảng thời gian chỉ định
        
        Args:
            start_date: Ngày bắt đầu (YYYY-MM-DD)
            end_date: Ngày kết thúc (YYYY-MM-DD)
            cloud_threshold: Ngưỡng mây tối đa (%)
            
        Returns:
            ee.ImageCollection: Collection ảnh Landsat đã lọc
        """
        collections = []
        
        for collection_id in self.config['satellite_data']['landsat']['collections']:
            collection = (ee.ImageCollection(collection_id)
                         .filterDate(start_date, end_date)
                         .filterBounds(self.study_area)
                         .filter(ee.Filter.lt('CLOUD_COVER', cloud_threshold)))
            collections.append(collection)
        
        # Merge tất cả collections
        merged_collection = collections[0]
        for i in range(1, len(collections)):
            merged_collection = merged_collection.merge(collections[i])
        
        self.logger.info(f"Tìm thấy {merged_collection.size().getInfo()} ảnh Landsat "
                        f"từ {start_date} đến {end_date}")
        
        return merged_collection
    
    def get_sentinel2_collection(self,
                                start_date: str,
                                end_date: str,
                                cloud_threshold: int = 20) -> ee.ImageCollection:
        """
        Lấy dữ liệu Sentinel-2 cho khoảng thời gian chỉ định
        
        Args:
            start_date: Ngày bắt đầu (YYYY-MM-DD)
            end_date: Ngày kết thúc (YYYY-MM-DD)
            cloud_threshold: Ngưỡng mây tối đa (%)
            
        Returns:
            ee.ImageCollection: Collection ảnh Sentinel-2 đã lọc
        """
        collection_id = self.config['satellite_data']['sentinel2']['collection']
        
        collection = (ee.ImageCollection(collection_id)
                     .filterDate(start_date, end_date)
                     .filterBounds(self.study_area)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)))
        
        self.logger.info(f"Tìm thấy {collection.size().getInfo()} ảnh Sentinel-2 "
                        f"từ {start_date} đến {end_date}")
        
        return collection
    
    def calculate_spectral_indices(self, image: ee.Image) -> ee.Image:
        """
        Tính toán các chỉ số phổ
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            ee.Image: Ảnh có thêm các band chỉ số phổ
        """
        # NDVI
        ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
        
        # MNDWI
        mndwi = image.normalizedDifference(['B3', 'B6']).rename('MNDWI')
        
        # NDBI
        ndbi = image.normalizedDifference(['B6', 'B5']).rename('NDBI')
        
        # EVI
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
                'NIR': image.select('B5'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }).rename('EVI')
        
        return image.addBands([ndvi, mndwi, ndbi, evi])
    
    def mask_clouds_landsat(self, image: ee.Image) -> ee.Image:
        """
        Loại bỏ mây cho ảnh Landsat sử dụng QA band
        
        Args:
            image: Ảnh Landsat
            
        Returns:
            ee.Image: Ảnh đã loại mây
        """
        qa = image.select('QA_PIXEL')
        
        # Bit masks cho cloud và cloud shadow
        cloud_bit = 1 << 3
        cloud_shadow_bit = 1 << 4
        
        # Mask clouds và cloud shadows
        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
               qa.bitwiseAnd(cloud_shadow_bit).eq(0))
        
        return image.updateMask(mask)
    
    def mask_clouds_sentinel2(self, image: ee.Image) -> ee.Image:
        """
        Loại bỏ mây cho ảnh Sentinel-2 sử dụng QA60 band
        
        Args:
            image: Ảnh Sentinel-2
            
        Returns:
            ee.Image: Ảnh đã loại mây
        """
        qa = image.select('QA60')
        
        # Bit 10 và 11 là cloud masks
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        
        # Mask clouds và cirrus
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
               qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        
        return image.updateMask(mask)
    
    def create_composite(self, 
                        collection: ee.ImageCollection,
                        method: str = 'median') -> ee.Image:
        """
        Tạo ảnh composite từ collection
        
        Args:
            collection: Collection ảnh đầu vào
            method: Phương pháp tạo composite ('median', 'mean', 'mosaic')
            
        Returns:
            ee.Image: Ảnh composite
        """
        if method == 'median':
            composite = collection.median()
        elif method == 'mean':
            composite = collection.mean()
        elif method == 'mosaic':
            composite = collection.mosaic()
        else:
            raise ValueError(f"Phương pháp không hỗ trợ: {method}")
        
        # Clip theo vùng nghiên cứu
        return composite.clip(self.study_area)
    
    def export_image_to_drive(self, 
                             image: ee.Image, 
                             description: str,
                             folder: str = 'dong_thap_analysis',
                             scale: int = 30) -> ee.batch.Task:
        """
        Xuất ảnh lên Google Drive
        
        Args:
            image: Ảnh cần xuất
            description: Mô tả file
            folder: Thư mục trên Drive
            scale: Độ phân giải (meters)
            
        Returns:
            ee.batch.Task: Task export
        """
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            scale=scale,
            region=self.study_area,
            maxPixels=1e13,
            crs='EPSG:4326'
        )
        
        task.start()
        self.logger.info(f"Bắt đầu export ảnh: {description}")
        
        return task
    
    def get_task_status(self) -> List[Dict]:
        """
        Kiểm tra trạng thái các task
        
        Returns:
            List[Dict]: Danh sách trạng thái tasks
        """
        tasks = ee.batch.Task.list()
        
        status_list = []
        for task in tasks:
            status_list.append({
                'id': task.id,
                'description': task.config.get('description', ''),
                'state': task.state,
                'creation_timestamp': task.creation_timestamp_ms
            })
        
        return status_list
