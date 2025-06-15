from ultralytics import YOLO
from pathlib import Path


model = YOLO("model.pt")


image_folder = Path("dataset/val/blue/")
model.predict(image_folder)
