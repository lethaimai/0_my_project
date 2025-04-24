import os
import cv2 
import torch
import sys 
sys.path.append('Depth-Anything-V2/')
from depth_anything_v2.dpt import DepthAnythingV2
import numpy as np
import open3d as o3d

model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
model.load_state_dict(torch.load('Depth-Anything-V2/depth_anything_v2_vitl.pth', map_location='cpu'))
model.eval()
model.cuda()



video = cv2.VideoCapture('videos/video_8.mp4')

if not video.isOpened():
    print("Error opening video file")
    exit()

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Process the frame (e.g., display it)
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    depth = model.infer_image(frame)    
    depth = model.infer_image(frame)
    pcd = o3d.geometry.PointCloud() 

    # Convert tensor to numpy if needed
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    # Normalize depth values to 0-255 range
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = np.uint8(depth_norm)

    # Apply colormap for better visualization
    depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)  # TURBO, JET, or INFERNO work well for depth

    # Show the colorized depth map
    cv2.imshow('Depth', depth_colormap)
    cv2.waitKey(0)

    # Create point cloud from depth map
    h, w = depth.shape
    
    # Create coordinate arrays for x and y
    y, x = np.mgrid[0:h, 0:w]
    
    # Approximate camera intrinsics (you may want to adjust these)
    fx = 600.0  # approximate focal length in x direction
    fy = 600.0  # approximate focal length in y direction
    cx = w / 2   # principal point x
    cy = h / 2   # principal point y
    
    # Create 3D points
    z = depth.flatten()
    x = ((x.flatten() - cx) * z) / fx
    y = ((y.flatten() - cy) * z) / fy
    
    # Stack coordinates to create point cloud
    points = np.stack((x, y, z), axis=-1)
    
    # Remove points with zero or invalid depth
    valid_depth = (z > 0) & ~np.isnan(z) & ~np.isinf(z)
    points = points[valid_depth]
    
    # Get colors from the RGB frame for the valid points
    if len(frame.shape) == 3:  # If frame is RGB
        colors = frame.reshape(-1, 3)[valid_depth] / 255.0  # Normalize to [0, 1]
        colors = colors[:, [2, 1, 0]]  # BGR to RGB
    
    # Create and visualize Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if len(frame.shape) == 3:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Optionally, downsample the point cloud for better performance
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd, coord_frame])
    break
    
    # Exit on 'q' key press
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break





