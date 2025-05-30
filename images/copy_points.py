import json
import os

# Define the directory path
directory = "/Users/cedric/0_my_project/images"

# Source JSON file
source_json = "image_341.json"
source_path = os.path.join(directory, source_json)

# Target JSON files (image_61.json to image_80.json)
target_jsons = [f"image_{i}.json" for i in range(342, 359)]

# Read the source JSON file
with open(source_path, 'r') as f:
    source_data = json.load(f)

# Extract points from source JSON
points = [shape for shape in source_data.get('shapes', []) if shape.get('shape_type') == 'point']

# Process each target JSON file
for target_json in target_jsons:
    target_path = os.path.join(directory, target_json)
    
    # Check if target JSON exists
    if not os.path.exists(target_path):
        print(f"Target file {target_json} does not exist. Skipping.")
        continue
    
    # Read the target JSON file
    with open(target_path, 'r') as f:
        target_data = json.load(f)
    
    # Preserve existing shapes and filter out any existing points to avoid duplicates
    target_data['shapes'] = [shape for shape in target_data.get('shapes', []) if shape.get('shape_type') != 'point']
    
    # Add the points from the source JSON
    target_data['shapes'].extend(points)
    
    # Write the updated data back to the target JSON file
    with open(target_path, 'w') as f:
        json.dump(target_data, f, indent=2)
    
    print(f"Updated {target_json} with points from {source_json}")

print("Processing complete.")