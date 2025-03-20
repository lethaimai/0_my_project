import os
import json

def convert_labelme_to_yolo(json_path, output_dir, class_names):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    
    # Get the corresponding image filename (without extension)
    image_filename = os.path.splitext(os.path.basename(data['imagePath']))[0]
    txt_filename = image_filename + '.txt'
    txt_path = os.path.join(output_dir, txt_filename)
    
    with open(txt_path, 'w') as f:
        for shape in data['shapes']:
            if shape['shape_type'] == 'rectangle':
                label = shape['label']
                found_point = None
                for shape in data['shapes']:
                    if shape['shape_type'] == 'point':
                        if shape['label'] == label:
                            found_point = shape
                            break
                
                if found_point is None:
                    print('labelled mid point not found')
                    continue
                if label not in class_names:
                    continue  # Skip unknown classes
                
                class_id = class_names.index(label)
                points = shape['points']
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # Normalize coordinates
                x_center = ((x1 + x2) / 2) / image_width
                y_center = ((y1 + y2) / 2) / image_height
                width = abs(x2 - x1) / image_width
                height = abs(y2 - y1) / image_height
                x,y = found_point['points'][0]/image_width, found_point['points'][1]/image_height
                
                # Write to file in YOLO format
                f.write(f"{class_id} {x_center} {y_center} {width} {height} {x} {y}\n")

# Define your class names
class_names = [
    'ear_1', 'ear_2', 'black_surface', 'slider', 'hatch_handle', 
    'multimeter_probe', 'multimeter_connector', 'blue_button', 
    'red_button', 'red_comp', 'black_hole'
]

# Paths
json_dir = 'json_files'  # Folder containing JSON files
output_dir = 'labels'  # Folder to save YOLO format labels

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all JSON files in the json_files folder
for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        json_path = os.path.join(json_dir, json_file)
        convert_labelme_to_yolo(json_path, output_dir, class_names)

print("Conversion completed. YOLO format labels saved in the 'labels' folder.")
