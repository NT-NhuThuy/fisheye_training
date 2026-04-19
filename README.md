# Fisheye Training

Repository nay gom 2 phan chinh:
- `training/`: chia dataset va train mo hinh detection
- `violation_speedlimit/`: pipeline suy luan vi pham toc do

## Cai dat
```bash
pip install -r requirements/training.txt
pip install -r requirements/inference.txt
```

## Huong dan nhanh
- Tao file dataset config rieng tu `training/configs/dataset.example.yaml`, hoac chay:
  `python training/split_dataset.py --data-dir /path/to/data_fisheye`
- Train YOLO11: `python training/scripts/train_yolo11.py`
- Train YOLO26: `python training/scripts/train_yolo26.py`
- Train YOLOv8: `python training/scripts/train_yolov8.py`
- Train RT-DETR: `python training/scripts/train_rtdetr.py`
- Chay suy luan: `python violation_speedlimit/src/main.py --source path/to/video.mp4`

## Ghi chu training
- Script train uu tien dung `training/configs/dataset_balanced.yaml` neu da tao bang `split_dataset.py`.
- Co the chi dinh GPU bang `YOLO_DEVICE`.
- Co the chi dinh weight goc bang `YOLO_MODEL_PATH`.
