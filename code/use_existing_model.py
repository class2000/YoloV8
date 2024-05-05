from ultralytics import YOLO

# Step 1: Initialize the model
model = YOLO()

# Step 2: Load the pretrained weights
model = model.load(weights='/Users/andreidanila/Downloads/plastic-bottles-DatasetNinja/runs/detect/train_run1/weights/best.pt')

# Step 3: Optionally, you can train the model further on your dataset
model.train(data='/Users/andreidanila/Downloads/plastic-bottles-DatasetNinja/scripts/config.yaml', epochs=50)

# Step 4: Perform detection using the trained/fine-tuned model
test_images_path = '/Users/andreidanila/Downloads/plastic-bottles-DatasetNinja/data/test/images'
results = model.predict(source=test_images_path)

# Step 5: Display and/or save the results
results.show()  # Show detections in a window
results.save()  # Save detected images

# Optionally, print detection results to console
print(results.xyxy)  # Prints bounding boxes, confidences, and class predictions
