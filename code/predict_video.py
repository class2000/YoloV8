import os
import cv2
from ultralytics import YOLO

# Define the directory for videos and models
VIDEOS_DIR = '/Users/andreidanila/Downloads/Plastic_Detection_DAV/videos'
MODEL_DIR = '/Users/andreidanila/Downloads/Plastic_Detection_DAV/code/runs/detect'

# Video files and model paths
video_filenames = [f'plastic{i}.mp4' for i in range(1, 9)]
model_epochs = ['50 epochs', '100 epochs', '200 epochs']

# Loop over each model
for epochs in model_epochs:
    model_path = os.path.join(MODEL_DIR, f'{epochs}/weights/best.pt')
    model = YOLO(model_path)  # Load the model

    # Loop over each video file
    for video_filename in video_filenames:
        video_path = os.path.join(VIDEOS_DIR, video_filename)
        video_path_out = os.path.join(VIDEOS_DIR, f'{video_filename.replace(".mp4", "")}_{epochs}_out.mp4')

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        # Get video properties
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Failed to read from the video file")

        H, W = frame.shape[:2]
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Initialize video writer with a more compatible codec
        out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'avc1'), fps, (W, H))

        # Set the detection threshold
        threshold = 0.5

        # Process each frame in the video
        while ret:
            results = model(frame)[0]  # Perform detection
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > threshold:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            out.write(frame)
            ret, frame = cap.read()

        # Release resources for this video
        cap.release()
        out.release()

cv2.destroyAllWindows()
