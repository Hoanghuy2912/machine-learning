"""
Module trực quan hóa bản đồ và biểu đồ cho phân tích biến động sử dụng đất
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import rasterio
import rasterio.plot
import geopandas as gpd
from rasterio.warp import transform_bounds
import contextily as ctx
from typing import Dict, List, Tuple, Optional, Union
import logging
import yaml
from pathlib import Path
import base64
from io import BytesIO

class MapVisualizer:
    """Class tạo bản đồ và biểu đồ trực quan"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Khởi tạo visualizer
        
        Args:
            config_path: Đường dẫn file cấu hình
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.land_cover_classes = self.config['land_cover_classes']
        self.colors = self._setup_colors()
        
        # Thiết lập style matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
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
    
    def _setup_colors(self) -> Dict:
        """Thiết lập bảng màu cho các lớp phủ đất"""
        colors = {}
        for class_key, class_info in self.land_cover_classes.items():
            colors[class_info['id']] = class_info['color']
        return colors
    
    def create_classification_map(self, 
                                 raster_path: str,
                                 output_path: str,
                                 title: str = "Bản đồ phân loại sử dụng đất",
                                 boundary_path: Optional[str] = None) -> str:
        """
        Tạo bản đồ phân loại sử dụng đất
        
        Args:
            raster_path: Đường dẫn file raster phân loại
            output_path: Đường dẫn file ảnh đầu ra
            title: Tiêu đề bản đồ
            boundary_path: Đường dẫn shapefile ranh giới (optional)
            
        Returns:
            str: Đường dẫn file ảnh đã tạo
        """
        # Đọc dữ liệu raster
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            bounds = src.bounds
            transform = src.transform
            crs = src.crs
        
        # Thiết lập figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Tạo colormap và norm
        unique_classes = np.unique(data[data > 0])
        colors_list = [self.colors.get(cls, '#000000') for cls in unique_classes]
        cmap = ListedColormap(colors_list)
        norm = BoundaryNorm(
            boundaries=np.arange(len(unique_classes) + 1) - 0.5,
            ncolors=len(unique_classes)
        )
        
        # Vẽ bản đồ
        im = ax.imshow(data, cmap=cmap, norm=norm, extent=[bounds.left, bounds.right, 
                                                          bounds.bottom, bounds.top])
        
        # Thêm ranh giới hành chính nếu có
        if boundary_path and Path(boundary_path).exists():
            boundary = gpd.read_file(boundary_path)
            if boundary.crs != crs:
                boundary = boundary.to_crs(crs)
            boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)
        
        # Tạo legend
        legend_elements = []
        for i, class_id in enumerate(unique_classes):
            class_name = self._get_class_name(class_id)
            legend_elements.append(
                patches.Patch(color=colors_list[i], label=class_name)
            )
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Định dạng bản đồ
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Kinh độ', fontsize=12)
        ax.set_ylabel('Vĩ độ', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Thêm north arrow và scale bar
        self._add_north_arrow(ax)
        self._add_scale_bar(ax, bounds)
        
        # Lưu bản đồ
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Đã tạo bản đồ phân loại: {output_path}")
        return output_path
    
    def create_change_map(self, 
                         raster_t1_path: str,
                         raster_t2_path: str,
                         output_path: str,
                         title: str = "Bản đồ biến động sử dụng đất") -> str:
        """
        Tạo bản đồ biến động giữa hai thời điểm
        
        Args:
            raster_t1_path: Đường dẫn raster thời điểm 1
            raster_t2_path: Đường dẫn raster thời điểm 2
            output_path: Đường dẫn file ảnh đầu ra
            title: Tiêu đề bản đồ
            
        Returns:
            str: Đường dẫn file ảnh đã tạo
        """
        # Đọc dữ liệu
        with rasterio.open(raster_t1_path) as src1:
            data_t1 = src1.read(1)
            bounds = src1.bounds
            crs = src1.crs
        
        with rasterio.open(raster_t2_path) as src2:
            data_t2 = src2.read(1)
        
        # Tạo bản đồ biến động
        change_map = np.where(data_t1 != data_t2, 1, 0)
        change_map[data_t1 == 0] = 0  # Loại bỏ NoData
        change_map[data_t2 == 0] = 0
        
        # Thiết lập figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Bản đồ thời điểm 1
        self._plot_single_map(ax1, data_t1, bounds, "Thời điểm 1")
        
        # Bản đồ thời điểm 2
        self._plot_single_map(ax2, data_t2, bounds, "Thời điểm 2")
        
        # Bản đồ biến động
        change_colors = ['white', 'red']
        change_cmap = ListedColormap(change_colors)
        im3 = ax3.imshow(change_map, cmap=change_cmap, extent=[bounds.left, bounds.right,
                                                              bounds.bottom, bounds.top])
        ax3.set_title("Khu vực biến động", fontsize=14, fontweight='bold')
        ax3.set_xlabel('Kinh độ')
        ax3.set_ylabel('Vĩ độ')
        ax3.grid(True, alpha=0.3)
        
        # Legend cho bản đồ biến động
        legend_elements = [
            patches.Patch(color='white', label='Không thay đổi'),
            patches.Patch(color='red', label='Có thay đổi')
        ]
        ax3.legend(handles=legend_elements, loc='upper right')
        
        # Tiêu đề chung
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Lưu bản đồ
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Đã tạo bản đồ biến động: {output_path}")
        return output_path
    
    def _plot_single_map(self, ax, data, bounds, title):
        """Vẽ một bản đồ đơn lẻ"""
        unique_classes = np.unique(data[data > 0])
        colors_list = [self.colors.get(cls, '#000000') for cls in unique_classes]
        cmap = ListedColormap(colors_list)
        norm = BoundaryNorm(
            boundaries=np.arange(len(unique_classes) + 1) - 0.5,
            ncolors=len(unique_classes)
        )
        
        im = ax.imshow(data, cmap=cmap, norm=norm, extent=[bounds.left, bounds.right,
                                                          bounds.bottom, bounds.top])
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Kinh độ')
        ax.set_ylabel('Vĩ độ')
        ax.grid(True, alpha=0.3)
        
        return im
    
    def _get_class_name(self, class_id: int) -> str:
        """Lấy tên lớp từ ID"""
        for class_key, class_info in self.land_cover_classes.items():
            if class_info['id'] == class_id:
                return class_info['name']
        return f"Unknown_{class_id}"
    
    def _add_north_arrow(self, ax):
        """Thêm mũi tên chỉ bắc"""
        ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
                   fontsize=14, fontweight='bold', ha='center', va='center')
        ax.annotate('↑', xy=(0.95, 0.90), xycoords='axes fraction',
                   fontsize=16, ha='center', va='center')
    
    def _add_scale_bar(self, ax, bounds):
        """Thêm thanh tỷ lệ"""
        # Tính độ dài thanh tỷ lệ (km)
        width = bounds.right - bounds.left
        scale_length = width * 0.2  # 20% chiều rộng bản đồ
        
        # Vị trí thanh tỷ lệ
        x_start = bounds.left + width * 0.05
        y_start = bounds.bottom + (bounds.top - bounds.bottom) * 0.05
        
        # Vẽ thanh tỷ lệ
        ax.plot([x_start, x_start + scale_length], [y_start, y_start], 
               'k-', linewidth=3)
        ax.text(x_start + scale_length/2, y_start - (bounds.top - bounds.bottom) * 0.02,
               f'{scale_length:.1f} km', ha='center', va='top', fontsize=10)
    
    def create_area_chart(self, 
                         area_stats: Dict[str, pd.DataFrame],
                         output_path: str,
                         chart_type: str = 'bar') -> str:
        """
        Tạo biểu đồ diện tích theo thời gian
        
        Args:
            area_stats: Dictionary thống kê diện tích theo năm
            output_path: Đường dẫn file ảnh đầu ra
            chart_type: Loại biểu đồ ('bar', 'line', 'stacked')
            
        Returns:
            str: Đường dẫn file ảnh đã tạo
        """
        # Chuẩn bị dữ liệu
        combined_data = []
        for year, stats in area_stats.items():
            for _, row in stats.iterrows():
                combined_data.append({
                    'Year': int(year),
                    'Class': row['Class_Name'],
                    'Area_ha': row['Area_ha'],
                    'Percentage': row['Percentage']
                })
        
        df = pd.DataFrame(combined_data)
        
        # Tạo biểu đồ
        if chart_type == 'bar':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Biểu đồ diện tích tuyệt đối
            pivot_area = df.pivot(index='Year', columns='Class', values='Area_ha')
            pivot_area.plot(kind='bar', ax=ax1, width=0.8)
            ax1.set_title('Diện tích các loại đất theo thời gian (ha)', fontsize=14)
            ax1.set_xlabel('Năm')
            ax1.set_ylabel('Diện tích (ha)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Biểu đồ tỷ lệ phần trăm
            pivot_percent = df.pivot(index='Year', columns='Class', values='Percentage')
            pivot_percent.plot(kind='bar', ax=ax2, width=0.8, stacked=True)
            ax2.set_title('Tỷ lệ các loại đất theo thời gian (%)', fontsize=14)
            ax2.set_xlabel('Năm')
            ax2.set_ylabel('Tỷ lệ (%)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
        elif chart_type == 'line':
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for class_name in df['Class'].unique():
                class_data = df[df['Class'] == class_name]
                ax.plot(class_data['Year'], class_data['Area_ha'], 
                       marker='o', linewidth=2, label=class_name)
            
            ax.set_title('Xu hướng thay đổi diện tích theo thời gian', fontsize=14)
            ax.set_xlabel('Năm')
            ax.set_ylabel('Diện tích (ha)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        elif chart_type == 'stacked':
            fig, ax = plt.subplots(figsize=(12, 8))
            
            pivot_area = df.pivot(index='Year', columns='Class', values='Area_ha')
            pivot_area.plot(kind='area', ax=ax, alpha=0.7)
            
            ax.set_title('Diện tích tích lũy các loại đất theo thời gian', fontsize=14)
            ax.set_xlabel('Năm')
            ax.set_ylabel('Diện tích (ha)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Đã tạo biểu đồ diện tích: {output_path}")
        return output_path
    
    def create_change_matrix_heatmap(self, 
                                   change_matrix: pd.DataFrame,
                                   output_path: str,
                                   title: str = "Ma trận biến động sử dụng đất") -> str:
        """
        Tạo heatmap ma trận biến động
        
        Args:
            change_matrix: Ma trận biến động
            output_path: Đường dẫn file ảnh đầu ra
            title: Tiêu đề
            
        Returns:
            str: Đường dẫn file ảnh đã tạo
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Tạo heatmap
        sns.heatmap(change_matrix, annot=True, fmt='d', cmap='YlOrRd',
                   ax=ax, cbar_kws={'label': 'Số pixel'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Đến (Thời điểm 2)', fontsize=12)
        ax.set_ylabel('Từ (Thời điểm 1)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Đã tạo heatmap ma trận biến động: {output_path}")
        return output_path
    
    def create_interactive_map(self, 
                             raster_path: str,
                             output_path: str,
                             boundary_path: Optional[str] = None) -> str:
        """
        Tạo bản đồ tương tác với Folium
        
        Args:
            raster_path: Đường dẫn file raster
            output_path: Đường dẫn file HTML đầu ra
            boundary_path: Đường dẫn shapefile ranh giới
            
        Returns:
            str: Đường dẫn file HTML đã tạo
        """
        # Đọc dữ liệu raster
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            bounds = src.bounds
            crs = src.crs
            
            # Chuyển đổi bounds sang WGS84
            if crs != 'EPSG:4326':
                bounds_wgs84 = transform_bounds(crs, 'EPSG:4326', *bounds)
            else:
                bounds_wgs84 = bounds
        
        # Tính center map
        center_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2
        center_lon = (bounds_wgs84[0] + bounds_wgs84[2]) / 2
        
        # Tạo base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Thêm raster layer (simplified - thực tế cần convert raster to image overlay)
        # Đây là placeholder, cần implement chi tiết hơn
        
        # Thêm ranh giới hành chính nếu có
        if boundary_path and Path(boundary_path).exists():
            boundary = gpd.read_file(boundary_path)
            if boundary.crs != 'EPSG:4326':
                boundary = boundary.to_crs('EPSG:4326')
            
            folium.GeoJson(
                boundary.to_json(),
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': 'red',
                    'weight': 2,
                    'fillOpacity': 0
                }
            ).add_to(m)
        
        # Thêm layer control
        folium.LayerControl().add_to(m)
        
        # Thêm fullscreen plugin
        plugins.Fullscreen().add_to(m)
        
        # Lưu map
        m.save(output_path)
        
        self.logger.info(f"Đã tạo bản đồ tương tác: {output_path}")
        return output_path
    
    def create_plotly_dashboard(self, 
                              analysis_results: Dict,
                              output_path: str) -> str:
        """
        Tạo dashboard tương tác chuyên nghiệp với Plotly
        
        Args:
            analysis_results: Kết quả phân tích
            output_path: Đường dẫn file HTML đầu ra
            
        Returns:
            str: Đường dẫn file HTML đã tạo
        """
        # Màu sắc chuyên nghiệp cho từng loại đất
        colors = {
            'Đất nông nghiệp': '#90EE90',
            'Đất đô thị': '#FF6B6B', 
            'Rừng': '#228B22',
            'Mặt nước': '#1E90FF',
            'Đất trống': '#DEB887'
        }
        
        # Tạo subplots với layout được cải thiện
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '📈 Xu hướng Diện tích Qua Thời Gian (ha)',
                '🔄 Tốc độ Thay đổi Hàng năm (%/năm)',
                '🗺️ Ma trận Biến động (pixel)',
                '🥧 Cơ cấu Sử dụng Đất 2025 (%)',
                '📊 So sánh 1990 vs 2025',
                '📉 Biến động Tích lũy'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"type": "heatmap"}, {"type": "pie"}],
                [{"colspan": 2}, None]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Biểu đồ đường xu hướng diện tích
        if 'area_statistics' in analysis_results:
            years = sorted(analysis_results['area_statistics'].keys())
            for class_name in colors.keys():
                areas = []
                for year in years:
                    stats = analysis_results['area_statistics'][year]
                    area = stats[stats['Class_Name'] == class_name]['Area_ha'].iloc[0]
                    areas.append(area)
                
                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=areas,
                        mode='lines+markers',
                        name=class_name,
                        line=dict(color=colors[class_name], width=3),
                        marker=dict(size=8),
                        hovertemplate=f'<b>{class_name}</b><br>' +
                                     'Năm: %{x}<br>' +
                                     'Diện tích: %{y:.1f} ha<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # 2. Biểu đồ cột tốc độ thay đổi
        if 'change_rates' in analysis_results:
            latest_period = list(analysis_results['change_rates'].keys())[-1]
            change_data = analysis_results['change_rates'][latest_period]
            
            bar_colors = [colors.get(name, '#888888') for name in change_data['Class_Name']]
            
            fig.add_trace(
                go.Bar(
                    x=change_data['Class_Name'],
                    y=change_data['Annual_change_rate'],
                    marker_color=bar_colors,
                    name=f'Giai đoạn {latest_period}',
                    hovertemplate='<b>%{x}</b><br>' +
                                 'Tốc độ: %{y:.1f}%/năm<br>' +
                                 '<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Ma trận biến động (heatmap)
        if 'change_matrices' in analysis_results:
            latest_matrix = list(analysis_results['change_matrices'].values())[-1]
            
            fig.add_trace(
                go.Heatmap(
                    z=latest_matrix.values,
                    x=latest_matrix.columns,
                    y=latest_matrix.index,
                    colorscale='YlOrRd',
                    showscale=True,
                    hovertemplate='Từ: %{y}<br>Đến: %{x}<br>Số pixel: %{z}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Biểu đồ tròn phân bố diện tích 2025
        if 'area_statistics' in analysis_results:
            latest_year = max(analysis_results['area_statistics'].keys())
            latest_stats = analysis_results['area_statistics'][latest_year]
            
            pie_colors = [colors.get(name, '#888888') for name in latest_stats['Class_Name']]
            
            fig.add_trace(
                go.Pie(
                    labels=latest_stats['Class_Name'],
                    values=latest_stats['Area_ha'],
                    marker_colors=pie_colors,
                    name=f"Năm {latest_year}",
                    hovertemplate='<b>%{label}</b><br>' +
                                 'Diện tích: %{value:.1f} ha<br>' +
                                 'Tỷ lệ: %{percent}<br>' +
                                 '<extra></extra>',
                    textinfo='label+percent',
                    textposition='auto'
                ),
                row=2, col=2
            )
        
        # 5. Biểu đồ so sánh 1990 vs 2025
        if 'area_statistics' in analysis_results:
            years_compare = [min(analysis_results['area_statistics'].keys()), 
                           max(analysis_results['area_statistics'].keys())]
            
            for i, year in enumerate(years_compare):
                stats = analysis_results['area_statistics'][year]
                
                fig.add_trace(
                    go.Bar(
                        x=stats['Class_Name'],
                        y=stats['Area_ha'],
                        name=f'Năm {year}',
                        marker_color=[colors.get(name, '#888888') for name in stats['Class_Name']],
                        opacity=0.7 if i == 0 else 1.0,
                        hovertemplate=f'<b>%{{x}} - {year}</b><br>' +
                                     'Diện tích: %{y:.1f} ha<extra></extra>'
                    ),
                    row=3, col=1
                )
        
        # Cập nhật layout với thiết kế chuyên nghiệp
        fig.update_layout(
            title={
                'text': "🌾 DASHBOARD PHÂN TÍCH BIẾN ĐỘNG SỬ DỤNG ĐẤT TỈNH ĐỒNG THÁP<br>" +
                       "<sub>Giai đoạn 1990-2025 | Công nghệ Viễn thám & Học máy</sub>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2E8B57'}
            },
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(240,248,255,0.8)',
            paper_bgcolor='white',
            height=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right", 
                x=1
            )
        )
        
        # Cập nhật trục với tiếng Việt
        fig.update_xaxes(title_text="Thời gian", row=1, col=1)
        fig.update_yaxes(title_text="Diện tích (ha)", row=1, col=1)
        
        fig.update_xaxes(title_text="Loại đất", row=1, col=2)
        fig.update_yaxes(title_text="Tốc độ (%/năm)", row=1, col=2)
        
        fig.update_xaxes(title_text="Đến (Loại đất)", row=2, col=1)
        fig.update_yaxes(title_text="Từ (Loại đất)", row=2, col=1)
        
        fig.update_xaxes(title_text="Loại đất", row=3, col=1)
        fig.update_yaxes(title_text="Diện tích (ha)", row=3, col=1)
        
        # Thêm annotations với thông tin quan trọng
        fig.add_annotation(
            text="📊 Nguồn: Dữ liệu mô phỏng demo<br>" +
                 "🛰️ Độ phân giải: 30m<br>" +
                 "🤖 ML Model: Random Forest",
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
        # Lưu dashboard với CSS tùy chỉnh
        html_string = fig.to_html(include_plotlyjs='cdn')
        
        # Thêm CSS để cải thiện giao diện
        custom_css = """
        <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
        }
        .plotly-graph-div {
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            padding: 20px;
            margin: 20px auto;
            max-width: 1400px;
        }
        </style>
        """
        
        html_string = html_string.replace('<head>', f'<head>{custom_css}')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_string)
        
        self.logger.info(f"Đã tạo dashboard tương tác: {output_path}")
        return output_path
    
    def create_time_series_animation(self, 
                                   raster_paths: List[str],
                                   years: List[int],
                                   output_path: str) -> str:
        """
        Tạo animation thay đổi theo thời gian
        
        Args:
            raster_paths: Danh sách đường dẫn raster theo thời gian
            years: Danh sách năm tương ứng
            output_path: Đường dẫn file GIF đầu ra
            
        Returns:
            str: Đường dẫn file GIF đã tạo
        """
        from matplotlib.animation import PillowWriter
        
        # Thiết lập figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Đọc dữ liệu đầu tiên để thiết lập colormap
        with rasterio.open(raster_paths[0]) as src:
            bounds = src.bounds
            
        # Tạo colormap chung
        all_data = []
        for path in raster_paths:
            with rasterio.open(path) as src:
                data = src.read(1)
                all_data.append(data)
        
        all_classes = np.unique(np.concatenate([d[d > 0] for d in all_data]))
        colors_list = [self.colors.get(cls, '#000000') for cls in all_classes]
        cmap = ListedColormap(colors_list)
        norm = BoundaryNorm(
            boundaries=np.arange(len(all_classes) + 1) - 0.5,
            ncolors=len(all_classes)
        )
        
        # Tạo animation
        writer = PillowWriter(fps=1)
        
        with writer.saving(fig, output_path, 100):
            for i, (path, year) in enumerate(zip(raster_paths, years)):
                ax.clear()
                
                with rasterio.open(path) as src:
                    data = src.read(1)
                
                im = ax.imshow(data, cmap=cmap, norm=norm, 
                             extent=[bounds.left, bounds.right, 
                                   bounds.bottom, bounds.top])
                
                ax.set_title(f'Sử dụng đất năm {year}', fontsize=16, fontweight='bold')
                ax.set_xlabel('Kinh độ')
                ax.set_ylabel('Vĩ độ')
                ax.grid(True, alpha=0.3)
                
                # Thêm legend
                if i == 0:  # Chỉ thêm legend cho frame đầu tiên
                    legend_elements = []
                    for j, class_id in enumerate(all_classes):
                        class_name = self._get_class_name(class_id)
                        legend_elements.append(
                            patches.Patch(color=colors_list[j], label=class_name)
                        )
                    ax.legend(handles=legend_elements, loc='center left', 
                             bbox_to_anchor=(1, 0.5))
                
                writer.grab_frame()
        
        plt.close()
        
        self.logger.info(f"Đã tạo animation: {output_path}")
        return output_path
