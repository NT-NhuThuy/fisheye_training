import os
from pathlib import Path

from ultralytics import YOLO


TRAIN_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = TRAIN_ROOT / "configs" / "dataset_balanced.yaml"
if not DATA_CONFIG_PATH.exists():
    DATA_CONFIG_PATH = TRAIN_ROOT / "configs" / "dataset.example.yaml"

DATA_CONFIG = str(DATA_CONFIG_PATH)
TRAIN_DEVICE = os.getenv("YOLO_DEVICE")
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "../rtdetr-l.pt")

model = YOLO(MODEL_PATH)

project_dir = "../result_train_fisheye"
run_name = "run_rtdetr"
os.makedirs(project_dir, exist_ok=True)

train_kwargs = dict(
    data=DATA_CONFIG,
    epochs=100,
    imgsz=960,
    batch=8,
    project=project_dir,
    name=run_name,
    patience=30,
    workers=8,
    mosaic=1.0,
    mixup=0.1,
    degrees=10.0,
    scale=0.5,
    fliplr=0.0,
)

if TRAIN_DEVICE:
    train_kwargs["device"] = TRAIN_DEVICE

model.train(**train_kwargs)
