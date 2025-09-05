#!/usr/bin/env python3
"""
Script hướng dẫn thiết lập Google Earth Engine
"""

import subprocess
import sys
import os

def check_gee_auth():
    """Kiểm tra xem GEE đã được xác thực chưa"""
    try:
        import ee
        ee.Initialize()
        print("[SUCCESS] Google Earth Engine đã được xác thực!")
        return True
    except Exception as e:
        print(f"[INFO] GEE chưa được xác thực: {str(e)}")
        return False

def authenticate_gee():
    """Hướng dẫn xác thực GEE"""
    print("\n" + "="*60)
    print("HƯỚNG DẪN XÁC THỰC GOOGLE EARTH ENGINE")
    print("="*60)
    
    print("\n1. Tạo tài khoản Google Earth Engine:")
    print("   - Truy cập: https://earthengine.google.com/")
    print("   - Đăng ký với tài khoản Google của bạn")
    print("   - Chờ Google phê duyệt (thường vài ngày)")
    
    print("\n2. Tạo Google Cloud Project:")
    print("   - Truy cập: https://console.cloud.google.com/")
    print("   - Tạo project mới hoặc sử dụng project có sẵn")
    print("   - Ghi nhớ Project ID")
    
    print("\n3. Xác thực EE CLI:")
    print("   Chạy lệnh sau và làm theo hướng dẫn:")
    print("   > earthengine authenticate")
    
    print("\n4. Thiết lập project:")
    print("   Thay YOUR_PROJECT_ID bằng Project ID của bạn:")
    print("   > earthengine set_project YOUR_PROJECT_ID")
    
    print("\n5. Kiểm tra:")
    print("   > python -c \"import ee; ee.Initialize(); print('OK')\"")
    
    print("\n" + "="*60)
    
    # Hỏi có muốn chạy tự động không
    response = input("\nBạn có muốn chạy 'earthengine authenticate' ngay bây giờ? (y/n): ")
    if response.lower() == 'y':
        try:
            subprocess.run(['earthengine', 'authenticate'], check=True)
            print("\n[SUCCESS] Đã hoàn thành xác thực!")
            
            # Hỏi về project
            project_id = input("\nNhập Project ID của bạn (để trống nếu chưa có): ")
            if project_id.strip():
                try:
                    subprocess.run(['earthengine', 'set_project', project_id], check=True)
                    print(f"[SUCCESS] Đã thiết lập project: {project_id}")
                except:
                    print(f"[WARNING] Không thể thiết lập project. Hãy chạy thủ công:")
                    print(f"earthengine set_project {project_id}")
            
        except subprocess.CalledProcessError:
            print("[ERROR] Lỗi khi chạy earthengine authenticate")
            print("Vui lòng chạy thủ công: earthengine authenticate")
        except FileNotFoundError:
            print("[ERROR] Không tìm thấy earthengine CLI")
            print("Vui lòng cài đặt: pip install earthengine-api")

def main():
    """Hàm main"""
    print("🌍 THIẾT LẬP GOOGLE EARTH ENGINE")
    print("="*50)
    
    # Kiểm tra xem earthengine đã được cài đặt chưa
    try:
        import ee
        print("[OK] earthengine-api đã được cài đặt")
    except ImportError:
        print("[ERROR] Chưa cài đặt earthengine-api")
        print("Chạy: pip install earthengine-api")
        return
    
    # Kiểm tra xác thực
    if check_gee_auth():
        print("\n[INFO] Google Earth Engine đã sẵn sàng sử dụng!")
        
        # Test một request đơn giản
        try:
            import ee
            ee.Initialize()
            point = ee.Geometry.Point([105.5, 10.8])  # Đồng Tháp
            image = ee.Image('COPERNICUS/S2_SR/20230101T032549_20230101T032546_T48PVS')
            print("[SUCCESS] Có thể truy cập dữ liệu Earth Engine!")
        except Exception as e:
            print(f"[WARNING] Có vẻ có vấn đề với quyền truy cập: {str(e)}")
    else:
        authenticate_gee()

if __name__ == "__main__":
    main()
