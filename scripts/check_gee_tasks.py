#!/usr/bin/env python3
"""
Script kiểm tra trạng thái các task Google Earth Engine
"""

import ee
from datetime import datetime

def check_tasks():
    """Kiểm tra trạng thái các task GEE"""
    
    try:
        ee.Initialize()
        print("🌍 TRẠNG THÁI GOOGLE EARTH ENGINE TASKS")
        print("=" * 50)
        
        # Lấy danh sách tasks
        tasks = ee.batch.Task.list()
        
        if not tasks:
            print("Không có task nào đang chạy")
            return
        
        # Lọc tasks của dự án Đồng Tháp
        dong_thap_tasks = [
            task for task in tasks 
            if 'dong_thap' in task.config.get('description', '').lower()
        ]
        
        if not dong_thap_tasks:
            print("Không có task Đồng Tháp nào")
            return
        
        print(f"Tìm thấy {len(dong_thap_tasks)} task của dự án Đồng Tháp:\n")
        
        # Hiển thị chi tiết từng task
        for i, task in enumerate(dong_thap_tasks[:10], 1):  # Chỉ hiển thị 10 task gần nhất
            description = task.config.get('description', 'Unknown')
            state = task.state
            
            # Tính thời gian
            creation_time = task.creation_timestamp_ms
            if creation_time:
                created = datetime.fromtimestamp(creation_time / 1000)
                time_str = created.strftime("%Y-%m-%d %H:%M:%S")
            else:
                time_str = "Unknown"
            
            # Icon trạng thái
            status_icons = {
                'READY': '⏳',
                'RUNNING': '🔄', 
                'COMPLETED': '✅',
                'FAILED': '❌',
                'CANCELLED': '⏹️'
            }
            
            icon = status_icons.get(state, '❓')
            
            print(f"{i:2d}. {icon} {description}")
            print(f"    Trạng thái: {state}")
            print(f"    Thời gian: {time_str}")
            
            if state == 'RUNNING':
                print(f"    📊 Đang xử lý...")
            elif state == 'COMPLETED':
                print(f"    📁 Có thể tải về từ Google Drive")
            elif state == 'FAILED':
                print(f"    ⚠️  Cần kiểm tra lỗi")
            
            print()
        
        # Thống kê tổng quan
        states = [task.state for task in dong_thap_tasks]
        state_counts = {state: states.count(state) for state in set(states)}
        
        print("📊 THỐNG KÊ TỔNG QUAN:")
        for state, count in state_counts.items():
            icon = status_icons.get(state, '❓')
            print(f"  {icon} {state}: {count} task(s)")
        
        # Hướng dẫn tiếp theo
        print("\n💡 HƯỚNG DẪN:")
        if any(state == 'COMPLETED' for state in states):
            print("✅ Một số task đã hoàn thành! Kiểm tra Google Drive.")
        if any(state == 'RUNNING' for state in states):
            print("🔄 Một số task đang chạy. Hãy đợi thêm.")
        if any(state == 'READY' for state in states):
            print("⏳ Một số task đang chờ xử lý.")
        
        print("\n🎯 Trong lúc chờ, bạn có thể:")
        print("  1. Chạy demo: python main.py --mode demo")
        print("  2. Chuẩn bị dữ liệu ground truth")
        print("  3. Kiểm tra lại sau 10-30 phút")
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        print("Vui lòng đảm bảo Google Earth Engine đã được xác thực")

if __name__ == "__main__":
    check_tasks()
