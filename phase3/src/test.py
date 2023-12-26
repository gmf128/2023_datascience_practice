from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train11/weights/best.pt')  # load a pretrained model (recommended for training)

#model.val(workers=0)
# test
results = model("../datasets/images/test", save_txt=True, save_conf=True, save=True)