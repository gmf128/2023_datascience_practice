from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)
# or (recommended):
# model = YOLO('yolov8n.pt')


# Train the model
results = model.train(data="..\datasets\data.yaml", epochs=200, imgsz=640, workers=0)

