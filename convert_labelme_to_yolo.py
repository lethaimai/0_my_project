import json
import os

# Define paths
project_dir = "/Users/cedric/0_my_project"
json_dir = os.path.join(project_dir, "images")
train_images_dir = os.path.join(project_dir, "dataset/images/train")
val_images_dir = os.path.join(project_dir, "dataset/images/val")
train_labels_dir = os.path.join(project_dir, "dataset/labels/train")
val_labels_dir = os.path.join(project_dir, "dataset/labels/val")

# Create label directories
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Class names from dataset.yaml
class_names = [
    "ear_1", "ear_2", "black_surface", "slider", "hatch_handle",
    "multimeter_probe", "multimeter_connector", "blue_button", "red_button", "red_comp",
    "black_hole", "center_black_hole", "center_red_comp", "center_red_button", "center_blue_button",
    "center_multimeter_connector", "center_multimeter_probe", "center_hatch_handle",
    "center_ear_2", "center_ear_1"
]
class_map = {name: idx for idx, name in enumerate(class_names)}

# Map center classes to their base classes
center_to_base = {
    "center_ear_1": "ear_1",
    "center_ear_2": "ear_2",
    "center_black_hole": "black_hole",
    "center_red_comp": "red_comp",
    "center_red_button": "red_button",
    "center_blue_button": "blue_button",
    "center_multimeter_connector": "multimeter_connector",
    "center_multimeter_probe": "multimeter_probe",
    "center_hatch_handle": "hatch_handle"
}
# Number of keypoints per object
num_keypoints = 2

def convert_to_yolo(json_path, train_images_dir, val_images_dir, train_labels_dir, val_labels_dir, class_map, num_keypoints):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}. Skipping.")
        return
    
    # Get image name and check if it's in train or val
    img_name = data.get('imagePath', '').split('/')[-1]
    if not img_name:
        print(f"Warning: No imagePath in {json_path}. Skipping.")
        return
    
    img_path_train = os.path.join(train_images_dir, img_name)
    img_path_val = os.path.join(val_images_dir, img_name)
    
    if os.path.exists(img_path_train):
        label_dir = train_labels_dir
    elif os.path.exists(img_path_val):
        label_dir = val_labels_dir
    else:
        print(f"Warning: Image {img_name} not found in dataset/images/train or val for {json_path}. Skipping.")
        return
    
    # Get image dimensions
    img_width = data.get('imageWidth', 640)
    img_height = data.get('imageHeight', 480)
    if img_width <= 0 or img_height <= 0:
        print(f"Warning: Invalid image dimensions in {json_path}. Using defaults (640x480).")
        img_width, img_height = 640, 480
    
    # Prepare YOLO label file
    label_name = os.path.splitext(img_name)[0] + '.txt'
    label_path = os.path.join(label_dir, label_name)
    
    # Group shapes by object (rectangles for bbox, points for keypoints)
    objects = {}
    for shape in data.get('shapes', []):
        label = shape.get('label')
        if label not in class_map:
            print(f"Warning: Label {label} in {json_path} not found in class_map. Skipping.")
            continue
        
        # Determine base class for keypoints
        base_label = center_to_base.get(label, label)
        base_class_id = class_map[base_label]
        objects.setdefault(base_class_id, {'bbox': None, 'keypoints': []})
        
        if shape['shape_type'] == 'rectangle' and label == base_label:
            if len(shape.get('points', [])) != 2:
                print(f"Warning: Invalid rectangle data in {json_path} for shape {label}. Skipping.")
                continue
            (x1, y1), (x2, y2) = shape['points']
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = abs(x2 - x1) / img_width
            height = abs(y2 - y1) / img_height
            objects[base_class_id]['bbox'] = [x_center, y_center, width, height]
        
        elif shape['shape_type'] == 'point' and label in center_to_base:
            if len(shape.get('points', [])) != 1:
                print(f"Warning: Invalid point data in {json_path} for shape {label}. Skipping.")
                continue
            x, y = shape['points'][0]
            objects[base_class_id]['keypoints'].append([x / img_width, y / img_height, 1])
    
    # Generate YOLO pose lines
    yolo_lines = []
    for class_id, obj in objects.items():
        if obj['bbox'] is None:
            print(f"Warning: No bounding box for class {class_names[class_id]} in {json_path}. Skipping.")
            continue
        bbox = obj['bbox']
        keypoints = obj['keypoints'][:num_keypoints]  # Take up to num_keypoints
        # Pad keypoints if fewer than num_keypoints
        kpt_data = []
        for i in range(num_keypoints):
            if i < len(keypoints):
                kpt_data.extend(keypoints[i])
            else:
                kpt_data.extend([0, 0, 0])  # Invisible keypoint
        line = [class_id] + bbox + kpt_data
        yolo_lines.append(' '.join(f"{x:.6f}" if isinstance(x, float) else f"{x}" for x in line))
    
    # Write YOLO label file
    if yolo_lines:
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        print(f"Created/Updated {label_path}")
    else:
        print(f"No valid annotations found in {json_path}. No label file created.")

# Process all JSON files in images directory
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
for json_file in json_files:
    convert_to_yolo(
        os.path.join(json_dir, json_file),
        train_images_dir,
        val_images_dir,
        train_labels_dir,
        val_labels_dir,
        class_map,
        num_keypoints
    )

print("Conversion complete.")