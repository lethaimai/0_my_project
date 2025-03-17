from ultralytics import YOLO
import cv2
import os

# Load the trained model
model_path = "runs/detect/yolo_custom_model/weights/best.pt"
model = YOLO(model_path)

# Video path
video_path = "videos/video_8.mp4"

# Output video path
output_path = "videos/output_video_8_slow.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = cap.get(cv2.CAP_PROP_FPS)

# Use a lower FPS for the output video (half the original speed)
output_fps = original_fps / 2  # Slows down the video to half speed
# If you want it even slower, use: output_fps = original_fps / 3 or / 4

# Initialize video writer with the slower FPS
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

# Process video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Run YOLOv8 inference on the frame
    results = model(frame)
    
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    # Write the frame to the output video
    out.write(annotated_frame)
    
    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video processing complete. Output saved to {output_path}")
print(f"Original video FPS: {original_fps}, Output video FPS: {output_fps}")