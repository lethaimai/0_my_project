from ultralytics import YOLO
import logging

# Configure logging to display in console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Print confirmation
print("Starting YOLO training process...")

# Load a pre-trained YOLO model
model = YOLO('yolov8n.pt')
print(f"Model loaded successfully: {model}")

# Train the model
results = model.train(
    data='/home/lethaimai/Desktop/master/0_my_project/dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='yolo_custom_model',
    verbose=True,  # Force verbose output
    device= 0
)

print("Training finished!")
print(f"Results: {results}")