from ultralytics import YOLO
import cv2
import os

# Paths
model_path = '/Users/cedric/0_my_project/runs/yolo_pose_model2/weights/best.pt'
video_dir = '/Users/cedric/0_my_project/videos'
output_dir = '/Users/cedric/0_my_project/runs/predict_videos'
video_files = ['video_7.mp4', 'video_8.mp4']

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load the trained model
print("Loading model...")
model = YOLO(model_path)
print("Model loaded successfully.")

# Process each video
for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    output_path = os.path.join(output_dir, f'processed_{video_file}')

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        continue

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing {video_file} ({total_frames} frames)...")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

        # Run inference on the frame
        results = model.predict(frame, verbose=False,conf=0.2)

        # Process results
        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy()  # Shape: (num_objects, num_keypoints, 2)
            for obj_idx, obj_kpts in enumerate(keypoints):
                for kpt_idx, (x, y) in enumerate(obj_kpts):
                    if x > 0 and y > 0:  # Only draw if keypoint is visible
                        # Draw keypoint as a circle
                        cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)

        # Write the frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Finished processing {video_file}. Output saved to {output_path}")

cv2.destroyAllWindows()
print("All videos processed.")