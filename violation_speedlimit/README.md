# Traffic Speed Violation Detection

He thong phat hien vi pham toc do tu video fisheye, su dung YOLO va ByteTrack.

## Cai dat
```bash
cd fisheye_training
pip install -r requirements/inference.txt
```

## Chay co ban
```bash
python violation_speedlimit/src/main.py --source path/to/video.mp4
```

## Cau truc
```text
violation_speedlimit/
├─ configs/
│  └─ my_tracker.yaml
├─ models/
│  └─ .gitkeep
└─ src/
   ├─ config.py
   ├─ main.py
   ├─ pipeline.py
   ├─ core/
   └─ output/
      ├─ annotator.py
      ├─ evidence_saver.py
      └─ report_logger.py
```
