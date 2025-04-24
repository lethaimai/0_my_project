from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Print confirmation
print("Starting YOLO pose estimation training...")

# Load a pre-trained YOLO pose model
model = YOLO('yolov8n-pose.pt')  # Pose model for keypoint detection
print(f"Model loaded successfully: {model}")

# Train the model
results = model.train(
    data='/Users/cedric/0_my_project/dataset/dataset.yaml',  # Path to dataset.yaml
    epochs=10,  # Reduced for testing; increase for full training
    imgsz=640,  # Image size
    batch=8,    # Batch size for CPU
    name='yolo_pose_model',  # Output directory name
    project='/Users/cedric/0_my_project/runs',  # Explicit output path
    verbose=True,  # Detailed logging
    device='mps',  # Use CPU (no GPU available)
    task='pose'    # Pose estimation task
)

print("Training finished!")
print(f"Results: {results}")