import os
from pathlib import Path
from ultralytics import YOLO

# 1. Tải mô hình YOLO11 Nano (Nhẹ nhất & nhanh nhất)
model = YOLO("../yolo11m.pt")  # Nằm trong PlugIR_Workspace

# 2. Định nghĩa thư mục lưu
project_dir = "../result_train_fisheye"
run_name = "run_yolo11"
os.makedirs(project_dir, exist_ok=True)

# 3. Tiến hành Training
model.train(
    data="/AIClub_NAS/core_baotg/thuyntn/Datasets/CV/data_fisheye/dataset.yaml",
    epochs=100,
    imgsz=960,             # Kích thước ảnh đầu vào. Ảnh 960 sẽ nét hơn 640 rất nhiều nhưng tốn RAM hơn
    batch=16,              # Với Nano YOLO có thể chạy Max batch 16 hoặc 32 tuỳ VRAM
    device=5,
    project=project_dir,
    name=run_name,
    patience=30,
    workers=8,

    # --- Các thông số Augmented Ảnh quan trọng ---
    mosaic=1.0,  # Ghép 4 ảnh đặc trưng, rất tốt để tìm biển báo nhỏ lấp ló
    mixup=0.0,   # Tắt mixup (trộn ảnh) giúp mô hình dễ học đặc trưng thô
    degrees=10.0,# Giới hạn xoay nhẹ
    scale=0.5,   # Phóng to/nhỏ tỉ lệ (rất quan trọng vì biển tốc độ có lúc xa lúc gần)
    fliplr=0.0,  # Tắt lật chiều dọc (tránh nhầm hướng mũi tên/hướng đường)
)
