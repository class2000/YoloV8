# YOLOv8 Plastic Bottle Detection

This project utilizes the YOLOv8 model to detect plastic bottles in images and video streams. It includes training a new model with the YOLOv8 architecture, and using pre-trained models for detection enhancement.

## Directory Structure

- **`use_new_model.py`**: Script to train a new YOLOv8 model using configuration settings specified in `config.yaml`.
- **`config.yaml`**: Contains paths to the dataset, training, and validation images, and specifies the class labels for detection.
- **`predict_video.py`**: Utilizes trained models to detect plastic bottles in video streams, iterating through models trained for different epochs to compare performance.
- **`use_existing_model.py`**: Script for loading and optionally further training a pre-existing model, then using it for prediction on new data.

## Setup Instructions

Before running the scripts, ensure the following steps are completed:

1. **Install Ultralytics YOLOv8**:
   - Install the YOLOv8 package from Ultralytics. Follow the installation guide on their official page appropriate for your operating system to avoid any compatibility issues.

2. **Prepare Your Environment**:
   - Ensure Python is installed along with necessary packages such as `opencv-python` for video processing.
   - Use `pip` to install the `ultralytics` package:
     ```
     pip install ultralytics
     ```

## Configuration (`config.yaml`)

- **Data Source**: Specify the source file for the dataset.
- **Paths**: Set relative paths to the train and validation datasets.
- **Classes**: Define the classes for detection. In this project, label `0` is assigned to bottles.

## Training the Model (`use_new_model.py`)

- Ensure all paths in `config.yaml` are correctly set up.
- Run the script to start training:
