#!/usr/bin/env python3
"""
Script thiết lập môi trường cho dự án phân tích biến động sử dụng đất
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_gee_authentication():
    """Thiết lập xác thực Google Earth Engine"""
    print("🌍 Thiết lập Google Earth Engine...")
    
    try:
        # Kiểm tra xem earthengine đã được cài đặt chưa
        import ee
        
        # Thử xác thực
        try:
            ee.Initialize()
            print("✅ Google Earth Engine đã được xác thực")
            return True
        except:
            print("⚠️ Cần xác thực Google Earth Engine")
            print("Vui lòng chạy lệnh sau để xác thực:")
            print("earthengine authenticate")
            return False
            
    except ImportError:
        print("❌ Chưa cài đặt earthengine-api")
        return False

def create_directories():
    """Tạo các thư mục cần thiết"""
    print("📁 Tạo cấu trúc thư mục...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/reference",
        "outputs/maps",
        "outputs/reports",
        "outputs/statistics",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Đã tạo thư mục: {directory}")

def check_gdal_installation():
    """Kiểm tra cài đặt GDAL"""
    print("🗺️ Kiểm tra GDAL...")
    
    try:
        import gdal
        print(f"✅ GDAL version: {gdal.__version__}")
        return True
    except ImportError:
        print("❌ GDAL chưa được cài đặt")
        print("Trên Windows: conda install -c conda-forge gdal")
        print("Trên Ubuntu: sudo apt-get install gdal-bin libgdal-dev")
        print("Trên macOS: brew install gdal")
        return False

def main():
    """Hàm main"""
    print("🚀 Thiết lập môi trường dự án Đồng Tháp Land Change Analysis")
    print("=" * 60)
    
    # Tạo thư mục
    create_directories()
    
    # Cài đặt requirements
    if not install_requirements():
        print("❌ Thiết lập thất bại")
        return
    
    # Kiểm tra GDAL
    check_gdal_installation()
    
    # Thiết lập GEE
    setup_gee_authentication()
    
    print("\n🎉 Thiết lập môi trường hoàn tất!")
    print("\nCác bước tiếp theo:")
    print("1. Xác thực Google Earth Engine (nếu chưa): earthengine authenticate")
    print("2. Chuẩn bị dữ liệu tham chiếu trong thư mục data/reference/")
    print("3. Chạy ứng dụng: python main.py")

if __name__ == "__main__":
    main()
