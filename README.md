# 🌾 Ứng dụng học máy đánh giá biến động sử dụng đất/lớp phủ bề mặt đất tỉnh Đồng Tháp

## 📋 Mô tả dự án
Ứng dụng nghiên cứu khoa học sử dụng công nghệ viễn thám và học máy để phân tích, đánh giá biến động sử dụng đất/lớp phủ bề mặt đất của tỉnh Đồng Tháp trong giai đoạn 1990-2025. Dự án cung cấp cái nhìn toàn diện về xu hướng thay đổi sử dụng đất và đưa ra các khuyến nghị cho quy hoạch bền vững.

## ✨ Tính năng chính
- 🛰️ **Xử lý ảnh viễn thám**: Tích hợp dữ liệu Landsat (1990-2020) và Sentinel-2 (2015-2025)
- 🤖 **Machine Learning**: Phân loại sử dụng đất với Random Forest, SVM, CNN
- 📊 **Phân tích biến động**: Phân tích không gian-thời gian chi tiết
- 🗺️ **Bản đồ chuyên đề**: Tạo bản đồ hiện trạng và biến động sử dụng đất
- 📈 **Trực quan hóa**: Dashboard tương tác và biểu đồ xu hướng
- 📄 **Báo cáo tự động**: Tạo báo cáo Excel, PDF với thống kê chi tiết

## 🎯 Mục tiêu nghiên cứu
1. **Xây dựng bản đồ hiện trạng** sử dụng đất cho các mốc thời gian 1990, 2000, 2010, 2020, 2025
2. **Ứng dụng học máy** để phân loại ảnh viễn thám với độ chính xác >85%
3. **Phân tích biến động** không gian-thời gian của các loại hình sử dụng đất
4. **Đưa ra khuyến nghị** cho công tác quy hoạch và phát triển bền vững

## 📊 Dữ liệu đầu vào
- **Ảnh viễn thám**: Landsat TM/ETM+/OLI (1990-2020, 30m), Sentinel-2 (2015-2025, 10m)
- **Ranh giới hành chính**: Tỉnh Đồng Tháp
- **Dữ liệu tham chiếu**: Ground truth, báo cáo đất đai
- **Dữ liệu phụ trợ**: DEM, NDVI, MNDWI, NDBI

## 🏗️ Cấu trúc dự án
```
dong_thap_land_change/
├── 📁 data/                      # Dữ liệu đầu vào
│   ├── raw/                     # Dữ liệu thô từ vệ tinh
│   ├── processed/               # Dữ liệu đã xử lý
│   └── reference/               # Dữ liệu tham chiếu, ground truth
├── 📁 src/                       # Mã nguồn chính
│   ├── data_processing/         # Xử lý dữ liệu viễn thám
│   │   ├── gee_client.py       # Google Earth Engine client
│   │   └── preprocessor.py     # Tiền xử lý ảnh
│   ├── models/                  # Mô hình Machine Learning
│   │   └── ml_models.py        # Random Forest, SVM, CNN
│   ├── analysis/                # Phân tích biến động
│   │   └── change_detection.py # Phân tích thay đổi sử dụng đất
│   └── visualization/           # Trực quan hóa
│       └── map_visualizer.py   # Tạo bản đồ và biểu đồ
├── 📁 notebooks/                 # Jupyter notebooks demo
├── 📁 outputs/                   # Kết quả đầu ra
│   ├── maps/                   # Bản đồ phân loại và biến động
│   ├── reports/                # Báo cáo Excel, PDF, Markdown
│   └── statistics/             # Thống kê và ma trận biến động
├── 📁 config/                    # Cấu hình dự án
│   └── config.yaml             # File cấu hình chính
├── 📁 scripts/                   # Scripts tiện ích
│   └── setup_environment.py    # Script thiết lập môi trường
├── main.py                      # Ứng dụng chính
└── run_analysis.py             # Script khởi chạy nhanh
```

## 🚀 Cài đặt và Sử dụng

### Yêu cầu hệ thống
- Python 3.7+
- GDAL
- Google Earth Engine account

### Cài đặt nhanh
```bash
# 1. Clone repository
git clone <repository-url>
cd dong_thap_land_change

# 2. Cài đặt dependencies
pip install -r requirements.txt

# 3. Thiết lập môi trường
python scripts/setup_environment.py

# 4. Chạy demo (không cần Google Earth Engine)
python main.py --mode demo

# 5. Thiết lập Google Earth Engine (cho dữ liệu thực)
python scripts/setup_gee.py

# 6. Chạy phân tích đầy đủ
python run_analysis.py
```

### Sử dụng chi tiết
```bash
# Chạy demo với dữ liệu mẫu (không cần GEE)
python main.py --mode demo

# Chạy toàn bộ quy trình
python main.py --mode full

# Chỉ tải dữ liệu từ Google Earth Engine
python main.py --mode download

# Chạy với cấu hình tùy chỉnh
python main.py --config custom_config.yaml
```

## 🚨 Xử lý lỗi thường gặp

### Lỗi Unicode trên Windows
```
UnicodeEncodeError: 'charmap' codec can't encode character
```
**Giải pháp**: Đã được sửa trong phiên bản mới, sử dụng UTF-8 encoding.

### Lỗi Google Earth Engine authentication
```
ee.Initialize: no project found
```
**Giải pháp**:
1. Chạy: `python scripts/setup_gee.py`
2. Làm theo hướng dẫn xác thực
3. Hoặc chạy demo: `python main.py --mode demo`

### Lỗi GDAL trên Windows
```
❌ GDAL chưa được cài đặt
```
**Giải pháp**:
- **Anaconda**: `conda install -c conda-forge gdal`
- **pip**: `pip install gdal` (có thể cần Visual Studio Build Tools)

### Lỗi memory khi xử lý ảnh lớn
**Giải pháp**: Giảm `chunk_size` trong `config/config.yaml`

## 📈 Quy trình phân tích

### 1. Thu thập dữ liệu
- Tải ảnh Landsat/Sentinel-2 từ Google Earth Engine
- Áp dụng cloud masking và atmospheric correction
- Tính toán các chỉ số phổ (NDVI, MNDWI, NDBI, EVI)

### 2. Tiền xử lý
- Cắt ảnh theo ranh giới tỉnh Đồng Tháp
- Resampling về cùng độ phân giải
- Chuẩn hóa dữ liệu

### 3. Huấn luyện mô hình
- Trích xuất training samples từ ground truth
- Huấn luyện Random Forest, SVM, CNN
- Đánh giá và lựa chọn mô hình tốt nhất

### 4. Phân loại ảnh
- Phân loại ảnh cho các năm 1990, 2000, 2010, 2020, 2025
- Tạo bản đồ sử dụng đất với 5 lớp chính:
  - 🌾 Đất nông nghiệp
  - 🏙️ Đất đô thị
  - 🌳 Rừng
  - 💧 Mặt nước
  - 🏜️ Đất trống

### 5. Phân tích biến động
- Tính toán ma trận biến động (Change Matrix)
- Phân tích xu hướng thay đổi diện tích
- Tính tốc độ thay đổi hàng năm
- Đánh giá landscape metrics

### 6. Trực quan hóa
- Bản đồ phân loại cho từng thời kỳ
- Bản đồ biến động giữa các giai đoạn
- Biểu đồ xu hướng diện tích
- Dashboard tương tác
- Animation thay đổi theo thời gian

## 🛠️ Công nghệ sử dụng

### Core Technologies
- **🐍 Python**: Ngôn ngữ lập trình chính
- **🌍 Google Earth Engine**: Xử lý ảnh viễn thám quy mô lớn
- **🗺️ GDAL/Rasterio**: Xử lý dữ liệu không gian
- **🧠 scikit-learn**: Machine Learning algorithms
- **🤖 TensorFlow**: Deep Learning cho CNN

### Visualization & Analysis
- **📊 Matplotlib/Seaborn**: Static plots và charts
- **📈 Plotly**: Interactive dashboards
- **🗺️ Folium**: Interactive maps
- **📋 Pandas**: Data manipulation
- **🔢 NumPy**: Numerical computing

### Geospatial Processing
- **🌐 GeoPandas**: Vector data processing
- **📐 Shapely**: Geometric operations
- **🗺️ Rasterio**: Raster data I/O
- **📍 Pyproj**: Coordinate transformations

## 📊 Kết quả kỳ vọng
- ✅ **Bộ bản đồ phân loại** Đồng Tháp 1990-2025 với độ chính xác >85%
- ✅ **Ma trận biến động** chi tiết cho từng giai đoạn
- ✅ **Báo cáo xu hướng** mở rộng đô thị, thay đổi nông nghiệp
- ✅ **Khuyến nghị chính sách** quy hoạch bền vững đến sau 2025

## 🌍 Ứng dụng thực tiễn
- 🏛️ **Quy hoạch sử dụng đất**: Hỗ trợ cơ quan quản lý tỉnh Đồng Tháp
- 🔬 **Nghiên cứu khoa học**: Dữ liệu cho nghiên cứu môi trường, nông nghiệp
- 🌡️ **Thích ứng biến đổi khí hậu**: Chính sách cho vùng Đồng bằng sông Cửu Long
- 📚 **Giáo dục**: Tài liệu học tập về GIS và Remote Sensing

## 📝 Ví dụ sử dụng

### Notebook Demo
```python
# Xem notebook hướng dẫn chi tiết
jupyter notebook notebooks/dong_thap_analysis_example.ipynb
```

### API Usage
```python
from src.data_processing.gee_client import GEEClient
from src.models.ml_models import RandomForestModel
from src.analysis.change_detection import LandUseChangeAnalyzer

# Khởi tạo components
gee_client = GEEClient()
rf_model = RandomForestModel()
analyzer = LandUseChangeAnalyzer()

# Tải và phân loại dữ liệu
composite = gee_client.create_composite(collection)
predictions = rf_model.predict(features)
change_matrix = analyzer.create_change_matrix(map1, map2)
```

## 🤝 Đóng góp
Chúng tôi hoan nghênh mọi đóng góp cho dự án! Vui lòng:
1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📜 License
Dự án được phát hành dưới MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 👥 Tác giả & Liên hệ
- **Nhóm nghiên cứu**: Phân tích Biến động Sử dụng Đất
- **Khu vực nghiên cứu**: Tỉnh Đồng Tháp, Việt Nam
- **Thời gian**: 2024

## 🙏 Lời cảm ơn
- Google Earth Engine team cho platform mạnh mẽ
- NASA/USGS cho dữ liệu Landsat
- ESA cho dữ liệu Sentinel-2
- Tỉnh Đồng Tháp cho sự hỗ trợ dữ liệu và thông tin

---
*Dự án này nhằm đóng góp vào việc quản lý bền vững tài nguyên đất đai và ứng phó với biến đổi khí hậu tại vùng Đồng bằng sông Cửu Long.*
"# machine-learning" 
"# machine-learning" 
