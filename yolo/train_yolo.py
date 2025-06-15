from ultralytics import YOLO

# Choose a model architecture (you can use 'yolov8n-cls.pt', 'yolov8s-cls.pt', etc.)
model = YOLO('yolov8s-cls.pt')  # n=Nano, s=Small, m=Medium, etc.

# Train the model
model.train(
    data='dataset',  # This should point to the root folder that contains train/ and val/
    epochs=30,
    imgsz=224,  # Typical image size for classification
    batch=64,
)

metrics = model.val()
print(metrics)

model.save("model.pt")
