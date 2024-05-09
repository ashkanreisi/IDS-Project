from ultralytics import YOLO
import yaml
import os

os.environ['WANDB_MODE'] = 'disabled'

model = YOLO('yolov8x.pt')
model.train(data="/home/zach/jupyter/bones/yolov9/data.yaml",
            cls=0.3878,
            scale=0.008593,
            shear=1.526,
            warmup_momentum=0.88,
            degrees=0,
            lrf=0.01,
            perspective=0,
            translate=0,
            warmup_epochs=5,
            weight_decay=0,
            device=0,
            imgsz=640,
            val=False,
            batch=24)
