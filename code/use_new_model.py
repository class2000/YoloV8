from ultralytics import YOLO # type: ignore

# Load a model
model = YOLO("yolov8n.yaml")

# Use the model
results = model.train(data="/Users/andreidanila/Downloads/Plastic_Detection_DAV/code/config.yaml", epochs=1)  # train the modelpy