#!/usr/bin/env python3
"""
Script khởi chạy nhanh phân tích biến động sử dụng đất tỉnh Đồng Tháp
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Hàm main để chạy phân tích"""
    
    print("🌾" + "="*60 + "🌾")
    print("   PHÂN TÍCH BIẾN ĐỘNG SỬ DỤNG ĐẤT TỈNH ĐỒNG THÁP")
    print("🌾" + "="*60 + "🌾")
    
    print("\n🚀 Bắt đầu thiết lập môi trường...")
    
    # Kiểm tra Python version
    if sys.version_info < (3, 7):
        print("❌ Cần Python 3.7 trở lên")
        return
    
    # Chạy setup
   
    
    # Chạy ứng dụng chính
    try:
        print("\n🎯 Khởi chạy ứng dụng phân tích...")
        result = subprocess.run([sys.executable, "main.py", "--mode", "full"], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            print("⚠️ Gặp lỗi khi chạy phân tích đầy đủ")
            
            # Kiểm tra xem có phải lỗi GEE không
            if "Google Earth Engine" in result.stderr or "ee.Initialize" in result.stderr:
                print("\n🎭 Chuyển sang chế độ DEMO (không cần Google Earth Engine)")
                print("Chế độ này sử dụng dữ liệu mẫu để minh họa quy trình phân tích")
                
                # Chạy demo mode
                demo_result = subprocess.run([sys.executable, "main.py", "--mode", "demo"])
                if demo_result.returncode == 0:
                    print("\n✨ Demo hoàn thành thành công!")
                    print("\nĐể chạy với dữ liệu thực, vui lòng:")
                    print("1. Chạy: python scripts/setup_gee.py")
                    print("2. Xác thực Google Earth Engine")
                    print("3. Chạy lại: python main.py --mode full")
                else:
                    print("❌ Lỗi khi chạy demo")
            else:
                print("❌ Lỗi khác khi chạy ứng dụng")
                print(f"Chi tiết: {result.stderr}")
        else:
            print(result.stdout)
            
    except FileNotFoundError:
        print("❌ Không tìm thấy main.py")
        return
    except KeyboardInterrupt:
        print("\n⏹️ Dừng bởi người dùng")
        return
    
    print("\n🎉 Hoàn thành phân tích!")
    print("\n📂 Kết quả được lưu trong thư mục 'outputs/'")
    print("📊 Xem báo cáo tại: outputs/reports/")
    print("🗺️ Xem bản đồ tại: outputs/maps/")
    print("📈 Xem dashboard tại: outputs/maps/dashboard.html")

if __name__ == "__main__":
    main()
