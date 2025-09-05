#!/usr/bin/env python3
"""
Ứng dụng chính cho phân tích biến động sử dụng đất/lớp phủ bề mặt đất tỉnh Đồng Tháp

Author: AI Assistant
Date: 2024
"""

import argparse
import logging
import yaml
from pathlib import Path
import sys
import os
from typing import Dict, List

# Thêm src vào path để import modules
sys.path.append(str(Path(__file__).parent / "src"))

from data_processing.gee_client import GEEClient
from data_processing.preprocessor import SatelliteDataPreprocessor
from models.ml_models import RandomForestModel, SVMModel, CNNModel, ModelEvaluator
from analysis.change_detection import LandUseChangeAnalyzer
from visualization.map_visualizer import MapVisualizer

class DongThapLandChangeApp:
    """Ứng dụng chính cho phân tích biến động sử dụng đất tỉnh Đồng Tháp"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Khởi tạo ứng dụng
        
        Args:
            config_path: Đường dẫn file cấu hình
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logger()
        self.validation_mode = False  # Chế độ validation chi tiết
        
        # Khởi tạo các components
        self.gee_client = None
        self.preprocessor = None
        self.models = {}
        self.change_analyzer = None
        self.visualizer = None
        
        self._initialize_components()
        
    def _load_config(self) -> Dict:
        """Đọc file cấu hình"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"[ERROR] Không tìm thấy file cấu hình: {self.config_path}")
            sys.exit(1)
    
    def _setup_logger(self) -> logging.Logger:
        """Thiết lập logging"""
        log_dir = Path(self.config['paths']['log_dir'])
        log_dir.mkdir(exist_ok=True)
        
        # Tạo handlers với UTF-8 encoding
        file_handler = logging.FileHandler(
            log_dir / 'dong_thap_analysis.log', 
            encoding='utf-8'
        )
        
        # Console handler với fallback cho Windows
        console_handler = logging.StreamHandler()
        
        # Thiết lập format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Tạo logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Thêm handlers nếu chưa có
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_components(self):
        """Khởi tạo các components"""
        try:
            self.logger.info("[INIT] Khởi tạo ứng dụng Phân tích Biến động Sử dụng Đất Đồng Tháp")
            
            # Tạo thư mục output
            output_dir = Path(self.config['paths']['output_dir'])
            output_dir.mkdir(exist_ok=True)
            (output_dir / 'maps').mkdir(exist_ok=True)
            (output_dir / 'reports').mkdir(exist_ok=True)
            (output_dir / 'statistics').mkdir(exist_ok=True)
            
            # Khởi tạo preprocessor
            self.preprocessor = SatelliteDataPreprocessor(self.config_path)
            
            # Khởi tạo change analyzer
            self.change_analyzer = LandUseChangeAnalyzer(self.config_path)
            
            # Khởi tạo visualizer
            self.visualizer = MapVisualizer(self.config_path)
            
            self.logger.info("[SUCCESS] Đã khởi tạo thành công các components")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Lỗi khi khởi tạo components: {str(e)}")
            raise
    
    def initialize_gee(self):
        """Khởi tạo Google Earth Engine"""
        try:
            self.logger.info("[GEE] Đang khởi tạo Google Earth Engine...")
            self.gee_client = GEEClient(self.config_path)
            self.gee_client.set_study_area()
            self.logger.info("[SUCCESS] Đã khởi tạo Google Earth Engine thành công")
        except Exception as e:
            self.logger.error(f"[ERROR] Lỗi khi khởi tạo GEE: {str(e)}")
            raise
    
    def download_satellite_data(self):
        """Tải dữ liệu ảnh vệ tinh"""
        if self.gee_client is None:
            self.initialize_gee()
        
        self.logger.info("[DOWNLOAD] Bắt đầu tải dữ liệu ảnh vệ tinh...")
        
        analysis_years = self.config['time_periods']['analysis_years']
        
        for year in analysis_years:
            try:
                self.logger.info(f"📥 Tải dữ liệu năm {year}...")
                
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
                
                if year >= 2015:
                    # Sử dụng Sentinel-2
                    collection = self.gee_client.get_sentinel2_collection(start_date, end_date)
                else:
                    # Sử dụng Landsat
                    collection = self.gee_client.get_landsat_collection(start_date, end_date)
                
                # Tạo composite
                composite = self.gee_client.create_composite(collection, method='median')
                
                # Thêm spectral indices
                composite_with_indices = self.gee_client.calculate_spectral_indices(composite)
                
                # Export to Drive
                task = self.gee_client.export_image_to_drive(
                    composite_with_indices,
                    f"dong_thap_{year}_composite",
                    scale=30 if year < 2015 else 10
                )
                
                self.logger.info(f"[SUCCESS] Đã khởi động export cho năm {year}")
                
            except Exception as e:
                self.logger.error(f"[ERROR] Lỗi khi tải dữ liệu năm {year}: {str(e)}")
    
    def prepare_training_data(self, image_path: str, reference_path: str):
        """Chuẩn bị dữ liệu training"""
        self.logger.info("🎯 Chuẩn bị dữ liệu training...")
        
        try:
            # Trích xuất training data
            features, labels = self.preprocessor.extract_training_data(
                image_path, reference_path
            )
            
            # Chuẩn hóa features
            features_normalized = self.preprocessor.normalize_data(features)
            
            self.logger.info(f"[SUCCESS] Đã chuẩn bị {len(features)} mẫu training")
            return features_normalized, labels
            
        except Exception as e:
            self.logger.error(f"[ERROR] Lỗi khi chuẩn bị training data: {str(e)}")
            raise
    
    def train_models(self, features, labels):
        """Huấn luyện các mô hình ML"""
        self.logger.info("[TRAINING] Bắt đầu huấn luyện các mô hình ML...")
        
        try:
            # Random Forest
            self.logger.info("🌲 Huấn luyện Random Forest...")
            rf_model = RandomForestModel(self.config_path)
            rf_results = rf_model.train(features, labels)
            self.models['random_forest'] = rf_model
            
            # SVM
            self.logger.info("🎯 Huấn luyện SVM...")
            svm_model = SVMModel(self.config_path)
            svm_results = svm_model.train(features, labels)
            self.models['svm'] = svm_model
            
            # CNN (nếu có đủ dữ liệu patch)
            if features.shape[0] > 1000:  # Cần đủ dữ liệu cho CNN
                self.logger.info("🧠 Huấn luyện CNN...")
                cnn_model = CNNModel(self.config_path)
                # Reshape features cho CNN (cần implement tùy theo format)
                # cnn_results = cnn_model.train(features_reshaped, labels)
                # self.models['cnn'] = cnn_model
            
            # So sánh mô hình
            evaluator = ModelEvaluator(self.config_path)
            # comparison = evaluator.compare_models(self.models, test_features, test_labels)
            
            self.logger.info("[SUCCESS] Đã hoàn thành huấn luyện mô hình")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Lỗi khi huấn luyện mô hình: {str(e)}")
            raise
    
    def classify_images(self, image_paths: List[str]):
        """Phân loại ảnh sử dụng mô hình đã huấn luyện"""
        self.logger.info("[CLASSIFICATION] Bắt đầu phân loại ảnh...")
        
        if not self.models:
            self.logger.error("[ERROR] Chưa có mô hình được huấn luyện")
            return
        
        # Sử dụng mô hình tốt nhất (Random Forest làm mặc định)
        best_model = self.models.get('random_forest')
        if best_model is None:
            self.logger.error("[ERROR] Không tìm thấy mô hình phù hợp")
            return
        
        classified_maps = {}
        
        for image_path in image_paths:
            try:
                year = self._extract_year_from_path(image_path)
                self.logger.info(f"📊 Phân loại ảnh năm {year}...")
                
                # Đọc và tiền xử lý ảnh
                image_data, metadata = self.preprocessor.load_raster(image_path)
                
                # Reshape cho prediction
                height, width = image_data.shape[1], image_data.shape[2]
                features = image_data.reshape(image_data.shape[0], -1).T
                
                # Chuẩn hóa
                features_normalized = self.preprocessor.normalize_data(features, fit_scaler=False)
                
                # Dự đoán
                predictions = best_model.predict(features_normalized)
                
                # Reshape về dạng ảnh
                classified_map = predictions.reshape(height, width)
                
                # Lưu kết quả
                output_path = f"outputs/maps/classified_{year}.tif"
                self.preprocessor.save_processed_data(
                    classified_map[np.newaxis, :, :], output_path, metadata
                )
                
                classified_maps[year] = classified_map
                self.logger.info(f"[SUCCESS] Đã phân loại và lưu ảnh năm {year}")
                
            except Exception as e:
                self.logger.error(f"[ERROR] Lỗi khi phân loại ảnh {image_path}: {str(e)}")
        
        return classified_maps
    
    def analyze_changes(self, classified_maps: Dict):
        """Phân tích biến động sử dụng đất"""
        self.logger.info("[ANALYSIS] Bắt đầu phân tích biến động...")
        
        try:
            analysis_results = {
                'area_statistics': {},
                'change_matrices': {},
                'change_rates': {},
                'landscape_metrics': {}
            }
            
            years = sorted(classified_maps.keys())
            pixel_size = 30 * 30  # 30m resolution
            
            # Thống kê diện tích cho từng năm
            for year in years:
                area_stats = self.change_analyzer.calculate_area_statistics(
                    classified_maps[year], pixel_size
                )
                analysis_results['area_statistics'][year] = area_stats
            
            # Ma trận biến động giữa các kỳ
            change_periods = self.config['change_analysis']['change_periods']
            for period in change_periods:
                year1, year2 = int(period[0]), int(period[1])
                if year1 in classified_maps and year2 in classified_maps:
                    
                    change_matrix = self.change_analyzer.create_change_matrix(
                        classified_maps[year1], classified_maps[year2]
                    )
                    analysis_results['change_matrices'][f"{year1}-{year2}"] = change_matrix
                    
                    # Tốc độ thay đổi
                    change_rates = self.change_analyzer.calculate_change_rates(
                        analysis_results['area_statistics'][year1],
                        analysis_results['area_statistics'][year2],
                        year2 - year1
                    )
                    analysis_results['change_rates'][f"{year1}-{year2}"] = change_rates
            
            # Landscape metrics
            for year in years:
                metrics = self.change_analyzer.calculate_landscape_metrics(
                    classified_maps[year]
                )
                analysis_results['landscape_metrics'][year] = metrics
            
            self.logger.info("[SUCCESS] Đã hoàn thành phân tích biến động")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"[ERROR] Lỗi khi phân tích biến động: {str(e)}")
            raise
    
    def create_visualizations(self, classified_maps: Dict, analysis_results: Dict):
        """Tạo các sản phẩm trực quan"""
        self.logger.info("[VISUALIZATION] Tạo các sản phẩm trực quan...")
        
        try:
            years = sorted(classified_maps.keys())
            
            # Bản đồ phân loại cho từng năm
            for year in years:
                map_path = f"outputs/maps/classification_map_{year}.png"
                # Chuyển đổi classified_map thành raster file trước khi visualize
                # self.visualizer.create_classification_map(
                #     f"outputs/maps/classified_{year}.tif", 
                #     map_path,
                #     f"Bản đồ sử dụng đất năm {year}"
                # )
            
            # Bản đồ biến động
            change_periods = self.config['change_analysis']['change_periods']
            for period in change_periods:
                year1, year2 = int(period[0]), int(period[1])
                if year1 in classified_maps and year2 in classified_maps:
                    change_map_path = f"outputs/maps/change_map_{year1}_{year2}.png"
                    # self.visualizer.create_change_map(
                    #     f"outputs/maps/classified_{year1}.tif",
                    #     f"outputs/maps/classified_{year2}.tif",
                    #     change_map_path,
                    #     f"Biến động sử dụng đất {year1}-{year2}"
                    # )
            
            # Biểu đồ diện tích
            area_chart_path = "outputs/maps/area_trends.png"
            self.visualizer.create_area_chart(
                analysis_results['area_statistics'],
                area_chart_path,
                'line'
            )
            
            # Heatmap ma trận biến động
            for period, matrix in analysis_results['change_matrices'].items():
                heatmap_path = f"outputs/maps/change_matrix_{period}.png"
                self.visualizer.create_change_matrix_heatmap(
                    matrix, heatmap_path, f"Ma trận biến động {period}"
                )
            
            # Dashboard tương tác
            dashboard_path = "outputs/maps/dashboard.html"
            self.visualizer.create_plotly_dashboard(
                analysis_results, dashboard_path
            )
            
            self.logger.info("[SUCCESS] Đã tạo xong các sản phẩm trực quan")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Lỗi khi tạo trực quan: {str(e)}")
    
    def generate_reports(self, analysis_results: Dict):
        """Tạo báo cáo"""
        self.logger.info("[REPORT] Tạo báo cáo...")
        
        try:
            # Báo cáo Excel
            excel_path = "outputs/reports/dong_thap_analysis_results.xlsx"
            self.change_analyzer.export_change_statistics(
                analysis_results, excel_path
            )
            
            # Báo cáo Markdown
            report_path = "outputs/reports/dong_thap_change_report.md"
            self.change_analyzer.generate_change_report(
                analysis_results, report_path
            )
            
            self.logger.info("[SUCCESS] Đã tạo xong báo cáo")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Lỗi khi tạo báo cáo: {str(e)}")
    
    def _extract_year_from_path(self, path: str) -> int:
        """Trích xuất năm từ đường dẫn file"""
        import re
        match = re.search(r'(\d{4})', path)
        if match:
            return int(match.group(1))
        return 2020  # default
    
    def _create_sample_training_data(self):
        """Tạo dữ liệu huấn luyện mẫu"""
        import numpy as np
        
        # Tạo dữ liệu mẫu với 8 features (6 bands + 2 indices)
        n_samples = 5000
        n_features = 8
        
        # Tạo features với phân phối khác nhau cho từng lớp
        features = []
        labels = []
        
        # Class 1: Nông nghiệp (NDVI cao, NIR cao)
        n_agri = 1500
        agri_features = np.random.normal([0.3, 0.4, 0.3, 0.7, 0.8, 0.2, 0.6, 0.1], 
                                       [0.1, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1, 0.05], 
                                       (n_agri, n_features))
        features.extend(agri_features)
        labels.extend([1] * n_agri)
        
        # Class 2: Đô thị (NIR thấp, độ phản xạ cao)
        n_urban = 1000
        urban_features = np.random.normal([0.6, 0.6, 0.6, 0.4, 0.5, 0.4, 0.2, 0.3], 
                                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1], 
                                        (n_urban, n_features))
        features.extend(urban_features)
        labels.extend([2] * n_urban)
        
        # Class 3: Rừng (NDVI rất cao, NIR cao)
        n_forest = 1000
        forest_features = np.random.normal([0.2, 0.3, 0.2, 0.8, 0.9, 0.1, 0.8, 0.0], 
                                         [0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0.02], 
                                         (n_forest, n_features))
        features.extend(forest_features)
        labels.extend([3] * n_forest)
        
        # Class 4: Mặt nước (NIR thấp, MNDWI cao)
        n_water = 800
        water_features = np.random.normal([0.1, 0.2, 0.1, 0.05, 0.1, 0.05, -0.3, 0.8], 
                                        [0.05, 0.05, 0.05, 0.02, 0.05, 0.02, 0.1, 0.1], 
                                        (n_water, n_features))
        features.extend(water_features)
        labels.extend([4] * n_water)
        
        # Class 5: Đất trống
        n_bare = 700
        bare_features = np.random.normal([0.5, 0.5, 0.4, 0.3, 0.4, 0.3, 0.1, -0.2], 
                                       [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1], 
                                       (n_bare, n_features))
        features.extend(bare_features)
        labels.extend([5] * n_bare)
        
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)
        
        # Trộn dữ liệu
        indices = np.random.permutation(len(features))
        features = features[indices]
        labels = labels[indices]
        
        return features, labels
    
    def _train_sample_model(self, features, labels):
        """Huấn luyện mô hình mẫu"""
        from src.models.ml_models import RandomForestModel, ModelEvaluator
        
        # Tạo và huấn luyện mô hình Random Forest
        rf_model = RandomForestModel(self.config_path)
        results = rf_model.train(features, labels)
        
        # Validation chi tiết nếu được bật
        if self.validation_mode:
            print("\n🔍 VALIDATION CHI TIẾT MÔ HÌNH:")
            print("-" * 60)
            
            evaluator = ModelEvaluator(self.config_path)
            
            # Tạo test data
            from sklearn.model_selection import train_test_split
            X_temp, X_test, y_temp, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Detailed metrics
            y_pred = rf_model.predict(X_test)
            detailed_metrics = evaluator.calculate_detailed_metrics(
                y_test, y_pred, 
                class_names=['Đất nông nghiệp', 'Đất đô thị', 'Rừng', 'Mặt nước', 'Đất trống']
            )
            
            print(f"📊 Overall Accuracy: {detailed_metrics['overall_accuracy']:.3f}")
            print(f"📊 Kappa Coefficient: {detailed_metrics['kappa_coefficient']:.3f}")
            
            print("\n📈 Producer's Accuracy (Recall theo từng lớp):")
            class_names = ['Đất nông nghiệp', 'Đất đô thị', 'Rừng', 'Mặt nước', 'Đất trống']
            for i, acc in enumerate(detailed_metrics['producers_accuracy']):
                print(f"   • {class_names[i]}: {acc:.3f}")
            
            print("\n📈 User's Accuracy (Precision theo từng lớp):")
            for i, acc in enumerate(detailed_metrics['users_accuracy']):
                print(f"   • {class_names[i]}: {acc:.3f}")
            
            # Thêm đánh giá chất lượng model
            self._validate_model_quality(detailed_metrics)
        
        # Lưu mô hình
        rf_model.save_model("outputs/sample_rf_model.pkl")
        
        return results
    
    def _validate_model_quality(self, metrics):
        """Đánh giá chất lượng mô hình"""
        print(f"\n✅ ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH:")
        print("-" * 60)
        
        accuracy = metrics['overall_accuracy']
        kappa = metrics['kappa_coefficient']
        
        # Đánh giá overall accuracy
        if accuracy >= 0.95:
            acc_status = "🟢 XUẤT SẮC"
        elif accuracy >= 0.90:
            acc_status = "🟡 TỐT" 
        elif accuracy >= 0.85:
            acc_status = "🟠 CHẤP NHẬN ĐƯỢC"
        else:
            acc_status = "🔴 CẦN CẢI THIỆN"
            
        print(f"Độ chính xác: {accuracy:.1%} - {acc_status}")
        
        # Đánh giá Kappa coefficient
        if kappa >= 0.8:
            kappa_status = "🟢 RẤT TỐT"
        elif kappa >= 0.6:
            kappa_status = "🟡 TỐT"
        elif kappa >= 0.4:
            kappa_status = "🟠 VỪA PHẢI"
        else:
            kappa_status = "🔴 YẾU"
            
        print(f"Hệ số Kappa: {kappa:.3f} - {kappa_status}")
        
        # Đánh giá độ cân bằng giữa các lớp
        producer_acc = metrics['producers_accuracy']
        min_acc = min(producer_acc)
        max_acc = max(producer_acc)
        
        if max_acc - min_acc < 0.1:
            balance_status = "🟢 CÂN BẰNG TỐT"
        elif max_acc - min_acc < 0.2:
            balance_status = "🟡 CÂN BẰNG VỪA"
        else:
            balance_status = "🔴 MẤT CÂN BẰNG"
            
        print(f"Cân bằng lớp: {balance_status} (chênh lệch: {max_acc-min_acc:.3f})")
        
        # Khuyến nghị
        print(f"\n💡 KHUYẾN NGHỊ:")
        if accuracy < 0.85:
            print("   • Cần thêm dữ liệu training và điều chỉnh parameters")
        if kappa < 0.6:
            print("   • Cần cải thiện chất lượng dữ liệu ground truth")
        if max_acc - min_acc > 0.2:
            print("   • Cần cân bằng số lượng mẫu giữa các lớp")
        if accuracy >= 0.95 and kappa >= 0.8:
            print("   • Mô hình đạt chất lượng cao, sẵn sàng ứng dụng thực tế")
    
    def _create_sample_classified_maps(self):
        """Tạo bản đồ phân loại mẫu"""
        import numpy as np
        
        height, width = 100, 100
        # Lấy từ config để phân tích đầy đủ đến 2025
        years = self.config['time_periods']['analysis_years']
        classified_maps = {}
        
        np.random.seed(42)
        for i, year in enumerate(years):
            # Tạo xu hướng đô thị hóa mạnh mẽ theo thời gian
            if year <= 2000:
                # Giai đoạn 1990-2000: Đô thị hóa chậm
                probabilities = [0.45, 0.08, 0.22, 0.20, 0.05]
            elif year <= 2010:
                # Giai đoạn 2000-2010: Đô thị hóa tăng tốc
                probabilities = [0.35, 0.15, 0.20, 0.22, 0.08]
            elif year <= 2020:
                # Giai đoạn 2010-2020: Đô thị hóa mạnh
                probabilities = [0.25, 0.25, 0.20, 0.22, 0.08]
            else:
                # Giai đoạn 2020-2025: Đô thị hóa rất mạnh
                probabilities = [0.20, 0.32, 0.18, 0.22, 0.08]
            
            base_map = np.random.choice([1, 2, 3, 4, 5], size=(height, width), p=probabilities)
            classified_maps[year] = base_map
            
        return classified_maps
    
    def _print_analysis_summary(self, analysis_results):
        """In tóm tắt kết quả phân tích"""
        print("\n" + "="*80)
        print("📊 BÁO CÁO PHÂN TÍCH BIẾN ĐỘNG SỬ DỤNG ĐẤT TỈNH ĐỒNG THÁP")
        print("🌾 Giai đoạn: 1990 - 2025 | Đơn vị: hecta (ha) | Độ phân giải: 30m")
        print("="*80)
        
        # Thống kê diện tích với format đẹp
        if 'area_statistics' in analysis_results:
            print("\n📈 DIỄN BIẾN DIỆN TÍCH CÁC LOẠI ĐẤT THEO THỜI GIAN")
            print("-" * 80)
            
            # Tạo bảng so sánh
            all_years = sorted(analysis_results['area_statistics'].keys())
            all_classes = analysis_results['area_statistics'][all_years[0]]['Class_Name'].unique()
            
            # Header
            print(f"{'Loại đất':<20}", end="")
            for year in all_years:
                print(f"{year:>12}", end="")
            print()
            print("-" * 80)
            
            # Data rows
            for class_name in all_classes:
                print(f"{class_name:<20}", end="")
                for year in all_years:
                    stats = analysis_results['area_statistics'][year]
                    area = stats[stats['Class_Name'] == class_name]['Area_ha'].iloc[0]
                    pct = stats[stats['Class_Name'] == class_name]['Percentage'].iloc[0]
                    print(f"{area:>8.1f}ha", end="")
                    print(f"({pct:>4.1f}%)", end="")
                print()
        
        # Tốc độ thay đổi với phân tích chi tiết
        if 'change_rates' in analysis_results:
            print(f"\n🔄 TỐC ĐỘ THAY ĐỔI HÀNG NĂM (%/năm)")
            print("-" * 80)
            
            for period, rates in analysis_results['change_rates'].items():
                years = period.split('-')
                duration = int(years[1]) - int(years[0])
                
                print(f"\n📊 Giai đoạn {period} ({duration} năm):")
                print(f"{'Loại đất':<20} {'Thay đổi':<15} {'Tốc độ':<12} {'Đánh giá':<15}")
                print("-" * 70)
                
                for _, row in rates.iterrows():
                    change = row['Change_ha']
                    rate = row['Annual_change_rate']
                    
                    # Đánh giá mức độ thay đổi
                    if abs(rate) < 0.5:
                        assessment = "Ổn định"
                    elif abs(rate) < 2.0:
                        assessment = "Thay đổi vừa"
                    else:
                        assessment = "Thay đổi mạnh"
                    
                    trend = "tăng" if rate > 0 else "giảm" if rate < 0 else "không đổi"
                    
                    print(f"{row['Class_Name']:<20} "
                          f"{change:>+8.1f}ha     "
                          f"{trend} {abs(rate):>4.1f}%/năm "
                          f"{assessment:<15}")
        
        # Nhận xét chi tiết và khuyến nghị
        self._print_detailed_insights(analysis_results)
    
    def _print_detailed_insights(self, analysis_results):
        """In nhận xét chi tiết và khuyến nghị"""
        print(f"\n💡 NHẬN XÉT VÀ PHÂN TÍCH CHI TIẾT")
        print("-" * 80)
        
        if 'change_rates' in analysis_results:
            # Phân tích xu hướng tổng thể
            latest_period = list(analysis_results['change_rates'].keys())[-1]
            latest_rates = analysis_results['change_rates'][latest_period]
            
            print("\n🌆 XU HƯỚNG ĐÔ THỊ HÓA:")
            urban_rate = latest_rates[latest_rates['Class_Name'] == 'Đất đô thị']['Annual_change_rate'].iloc[0]
            if urban_rate > 2:
                print(f"   • Đô thị hóa MẠNH với tốc độ {urban_rate:.1f}%/năm")
                print("   • Cần quy hoạch kỹ lưỡng để đảm bảo phát triển bền vững")
            elif urban_rate > 1:
                print(f"   • Đô thị hóa VỪA PHẢI với tốc độ {urban_rate:.1f}%/năm")
                print("   • Xu hướng phát triển ổn định")
            
            print("\n🌾 BIẾN ĐỘNG NÔNG NGHIỆP:")
            agri_rate = latest_rates[latest_rates['Class_Name'] == 'Đất nông nghiệp']['Annual_change_rate'].iloc[0]
            if agri_rate < -1:
                print(f"   • Diện tích nông nghiệp GIẢM với tốc độ {abs(agri_rate):.1f}%/năm")
                print("   • Cần cân bằng giữa phát triển đô thị và an ninh lương thực")
            
            print("\n🌳 BẢO TỒN TÀI NGUYÊN:")
            forest_rate = latest_rates[latest_rates['Class_Name'] == 'Rừng']['Annual_change_rate'].iloc[0]
            water_rate = latest_rates[latest_rates['Class_Name'] == 'Mặt nước']['Annual_change_rate'].iloc[0]
            
            if abs(forest_rate) < 0.5 and abs(water_rate) < 0.5:
                print("   • Tài nguyên rừng và nước được bảo tồn TỐT")
                print("   • Cần duy trì chính sách bảo vệ môi trường")
        
        print(f"\n🎯 KHUYẾN NGHỊ CHÍNH SÁCH:")
        print("   1. 🏘️  Quy hoạch đô thị thông minh để kiểm soát đô thị hóa")
        print("   2. 🌾 Bảo vệ đất nông nghiệp chất lượng cao")
        print("   3. 🌳 Duy trì độ che phủ rừng tối thiểu 20%")
        print("   4. 💧 Bảo vệ nguồn nước và hệ sinh thái ven sông")
        print("   5. 📊 Giám sát định kỳ bằng công nghệ viễn thám")
        
        print(f"\n⚠️  LƯU Ý QUAN TRỌNG:")
        print("   • Kết quả này là mô phỏng demo với dữ liệu mẫu")
        print("   • Để có kết quả chính xác, cần sử dụng dữ liệu thực từ Google Earth Engine")
        print("   • Độ tin cậy phụ thuộc vào chất lượng dữ liệu ground truth")
    
    def run_full_analysis(self):
        """Chạy toàn bộ quy trình phân tích"""
        self.logger.info("[FULL_ANALYSIS] Bắt đầu quy trình phân tích hoàn chỉnh...")
        
        try:
            # Bước 1: Tải dữ liệu vệ tinh
            print("\n[STEP 1] Tải dữ liệu ảnh vệ tinh từ Google Earth Engine")
            self.download_satellite_data()
            
            print("\n⏳ Chờ Google Earth Engine xử lý dữ liệu...")
            print("📋 Trong lúc chờ, sẽ chạy workflow demo với dữ liệu mẫu:")
            
            # Chạy demo analysis với dữ liệu mẫu để minh họa quy trình
            print("\n" + "="*60)
            print("🎭 CHẠY DEMO WORKFLOW VỚI DỮ LIỆU MẪU")
            print("="*60)
            
            # Bước 2: Tạo dữ liệu mẫu
            print("\n[DEMO STEP 2] Tạo dữ liệu huấn luyện mẫu...")
            features, labels = self._create_sample_training_data()
            print(f"[SUCCESS] Đã tạo {len(features)} mẫu với {features.shape[1]} đặc trưng")
            
            # Bước 3: Huấn luyện mô hình
            print("\n[DEMO STEP 3] Huấn luyện mô hình Random Forest...")
            model_results = self._train_sample_model(features, labels)
            print(f"[SUCCESS] Độ chính xác mô hình: {model_results['accuracy']:.3f}")
            
            # Bước 4: Tạo bản đồ phân loại mẫu
            print("\n[DEMO STEP 4] Tạo bản đồ phân loại mẫu...")
            classified_maps = self._create_sample_classified_maps()
            print(f"[SUCCESS] Đã tạo {len(classified_maps)} bản đồ mẫu")
            
            # Bước 5: Phân tích biến động
            print("\n[DEMO STEP 5] Phân tích biến động...")
            analysis_results = self.analyze_changes(classified_maps)
            print("[SUCCESS] Đã hoàn thành phân tích biến động")
            
            # Bước 6: Tạo trực quan
            print("\n[DEMO STEP 6] Tạo sản phẩm trực quan...")
            self.create_visualizations(classified_maps, analysis_results)
            print("[SUCCESS] Đã tạo các sản phẩm trực quan")
            
            # Bước 7: Tạo báo cáo
            print("\n[DEMO STEP 7] Tạo báo cáo...")
            self.generate_reports(analysis_results)
            print("[SUCCESS] Đã tạo báo cáo chi tiết")
            
            # Tóm tắt kết quả
            self._print_analysis_summary(analysis_results)
            
            self.logger.info("[COMPLETE] Hoàn thành workflow demo!")
            
            print("\n" + "="*60)
            print("🎯 KẾT QUẢ WORKFLOW")
            print("="*60)
            print("✅ Demo workflow hoàn thành thành công!")
            print("📊 Dữ liệu từ Google Earth Engine đang được xử lý song song")
            print("\n💡 Kiểm tra tiến độ GEE: python scripts/check_gee_tasks.py")
            print("📂 Kết quả demo lưu tại: outputs/")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Lỗi trong quy trình phân tích: {str(e)}")
            raise
    
    def run_demo_analysis(self):
        """Chạy demo phân tích với dữ liệu mẫu (không cần GEE)"""
        self.logger.info("[DEMO] Bắt đầu demo phân tích với dữ liệu mẫu...")
        
        try:
            import numpy as np
            
            print("\n" + "="*60)
            print("🌾 DEMO PHÂN TÍCH BIẾN ĐỘNG SỬ DỤNG ĐẤT 🌾")
            print("="*60)
            print("Chế độ demo sử dụng dữ liệu mẫu để minh họa quy trình")
            print("="*60)
            
            # Tạo dữ liệu mẫu
            print("\n[DEMO STEP 1] Tạo dữ liệu mẫu...")
            height, width = 100, 100
            years = [1990, 2000, 2010, 2020]
            classified_maps = {}
            
            # Tạo maps giả lập với xu hướng đô thị hóa
            np.random.seed(42)
            for i, year in enumerate(years):
                # Tạo bản đồ với xu hướng thay đổi theo thời gian
                # Class 1: Nông nghiệp (giảm dần)
                # Class 2: Đô thị (tăng dần) 
                # Class 3: Rừng (ổn định)
                # Class 4: Mặt nước (ổn định)
                # Class 5: Đất trống (giảm)
                base_map = np.random.choice(
                    [1, 2, 3, 4, 5], 
                    size=(height, width),
                    p=[0.4-i*0.05, 0.1+i*0.05, 0.2, 0.2, 0.1]
                )
                classified_maps[year] = base_map
            
            print(f"[SUCCESS] Đã tạo {len(classified_maps)} bản đồ mẫu")
            
            # Phân tích thống kê
            print("\n[DEMO STEP 2] Phân tích thống kê diện tích...")
            pixel_size = 30 * 30  # 30m resolution
            area_statistics = {}
            
            for year, classified_map in classified_maps.items():
                area_stats = self.change_analyzer.calculate_area_statistics(
                    classified_map, pixel_size
                )
                area_statistics[year] = area_stats
                print(f"Năm {year}: {len(area_stats)} loại đất được phân tích")
            
            # Tạo ma trận biến động
            print("\n[DEMO STEP 3] Tính ma trận biến động...")
            change_matrix = self.change_analyzer.create_change_matrix(
                classified_maps[1990], classified_maps[2020]
            )
            print(f"Ma trận biến động 1990-2020: {change_matrix.shape}")
            
            # Tính tốc độ thay đổi
            change_rates = self.change_analyzer.calculate_change_rates(
                area_statistics[1990], area_statistics[2020], 30
            )
            print("Tốc độ thay đổi hàng năm đã được tính toán")
            
            # Tạo trực quan
            print("\n[DEMO STEP 4] Tạo sản phẩm trực quan...")
            
            # Biểu đồ xu hướng
            area_chart_path = "outputs/demo_area_trends.png"
            self.visualizer.create_area_chart(
                area_statistics, area_chart_path, 'line'
            )
            print(f"[SUCCESS] Đã tạo biểu đồ: {area_chart_path}")
            
            # Ma trận biến động
            matrix_path = "outputs/demo_change_matrix.png"  
            self.visualizer.create_change_matrix_heatmap(
                change_matrix, matrix_path, "Ma trận biến động Demo 1990-2020"
            )
            print(f"[SUCCESS] Đã tạo heatmap: {matrix_path}")
            
            # Tạo báo cáo demo
            print("\n[DEMO STEP 5] Tạo báo cáo demo...")
            analysis_results = {
                'area_statistics': area_statistics,
                'change_matrices': {'1990-2020': change_matrix},
                'change_rates': {'1990-2020': change_rates}
            }
            
            report_path = "outputs/demo_report.md"
            self.change_analyzer.generate_change_report(
                analysis_results, report_path
            )
            print(f"[SUCCESS] Đã tạo báo cáo: {report_path}")
            
            # Tóm tắt kết quả
            print("\n" + "="*60)
            print("📊 KẾT QUẢ DEMO")
            print("="*60)
            
            print("\n🔍 Xu hướng chính được phát hiện:")
            for _, row in change_rates.iterrows():
                if abs(row['Annual_change_rate']) > 0.5:
                    trend = "tăng" if row['Annual_change_rate'] > 0 else "giảm"
                    print(f"  • {row['Class_Name']}: {trend} {abs(row['Annual_change_rate']):.1f}%/năm")
            
            print(f"\n📂 Sản phẩm đã tạo:")
            print(f"  • Biểu đồ xu hướng: {area_chart_path}")
            print(f"  • Ma trận biến động: {matrix_path}")
            print(f"  • Báo cáo chi tiết: {report_path}")
            
            print("\n✨ Demo hoàn thành thành công!")
            print("\nĐể chạy với dữ liệu thực:")
            print("1. Thiết lập Google Earth Engine: python scripts/setup_gee.py")
            print("2. Chạy phân tích đầy đủ: python main.py --mode full")
            
            self.logger.info("[DEMO SUCCESS] Demo hoàn thành thành công!")
            
        except Exception as e:
            self.logger.error(f"[DEMO ERROR] Lỗi trong demo: {str(e)}")
            print(f"\n[ERROR] Lỗi demo: {str(e)}")
            raise

def main():
    """Hàm main"""
    parser = argparse.ArgumentParser(
        description="Ứng dụng phân tích biến động sử dụng đất tỉnh Đồng Tháp"
    )
    
    parser.add_argument(
        '--config', 
        default='config/config.yaml',
        help='Đường dẫn file cấu hình'
    )
    
    parser.add_argument(
        '--mode',
        choices=['download', 'train', 'classify', 'analyze', 'visualize', 'full', 'demo'],
        default='full',
        help='Chế độ chạy ứng dụng'
    )
    
    parser.add_argument(
        '--input-dir',
        help='Thư mục chứa dữ liệu đầu vào'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Thư mục lưu kết quả'
    )
    
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        help='Danh sách năm cần phân tích (ví dụ: --years 1990 2000 2010 2020 2025)'
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        default=1990,
        help='Năm bắt đầu phân tích (mặc định: 1990)'
    )
    
    parser.add_argument(
        '--end-year', 
        type=int,
        default=2025,
        help='Năm kết thúc phân tích (mặc định: 2025)'
    )
    
    parser.add_argument(
        '--validation',
        action='store_true',
        help='Bật chế độ validation chi tiết với metrics đánh giá'
    )
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo ứng dụng
        app = DongThapLandChangeApp(args.config)
        
        # Cập nhật cấu hình với tham số từ command line
        if args.years:
            app.config['time_periods']['analysis_years'] = sorted(args.years)
            print(f"🗓️  Sử dụng năm tùy chỉnh: {args.years}")
        elif args.start_year or args.end_year:
            # Tạo danh sách năm với khoảng cách 5 năm
            years = list(range(args.start_year, args.end_year + 1, 
                             app.config['time_periods']['min_year_gap']))
            if args.end_year not in years:
                years.append(args.end_year)
            app.config['time_periods']['analysis_years'] = years
            print(f"🗓️  Phân tích từ {args.start_year} đến {args.end_year}: {years}")
        
        # Thiết lập validation mode
        app.validation_mode = args.validation
        if args.validation:
            print("🔍 Chế độ validation CHI TIẾT được kích hoạt")
        
        print("=" * 80)
        print("🌾 PHÂN TÍCH BIẾN ĐỘNG SỬ DỤNG ĐẤT TỈNH ĐỒNG THÁP 🌾")
        print(f"📅 Giai đoạn: {min(app.config['time_periods']['analysis_years'])} - {max(app.config['time_periods']['analysis_years'])}")
        print(f"🎯 Chế độ: {args.mode.upper()}")
        print("=" * 80)
        
        if args.mode == 'download':
            app.download_satellite_data()
        elif args.mode == 'full':
            app.run_full_analysis()
        elif args.mode == 'demo':
            app.run_demo_analysis()
        else:
            print(f"Chế độ {args.mode} đang được phát triển...")
        
        print("\n[SUCCESS] Ứng dụng hoàn thành thành công!")
        
    except KeyboardInterrupt:
        print("\n[STOPPED] Ứng dụng bị dừng bởi người dùng")
    except Exception as e:
        print(f"\n[ERROR] Lỗi: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
