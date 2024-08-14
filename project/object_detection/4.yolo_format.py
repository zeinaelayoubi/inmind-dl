import os
import json
import numpy as np
from PIL import Image

# Define directories
image_dir = 'images/'
npy_dir = 'npy_bboxes/'
json_dir = 'json_labels/'
combined_yolo_file = 'combined_annotations.txt'

# Function to get image dimensions
def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size  # Returns (width, height)

# Convert bounding box to YOLO format
def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    bbox_semantic_id, x_min, y_min, x_max, y_max, _ = bbox
    center_x = (x_min + x_max) / 2 / img_width
    center_y = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return [bbox_semantic_id, center_x, center_y, width, height]

# Process .npy files
def process_npy_file(npy_file, image_dims, combined_file):
    index = os.path.splitext(os.path.basename(npy_file))[0].split('_')[-1]
    image_filename = f'rgb_{index}.png'
    image_path = os.path.join(image_dir, image_filename)
    img_width, img_height = image_dims.get(image_filename, (1024, 768))

    annotations = np.load(npy_file)
    yolo_annotations = [convert_bbox_to_yolo_format(bbox, img_width, img_height) for bbox in annotations]

    with open(combined_file, 'a') as f:
        for annotation in yolo_annotations:
            f.write(' '.join(map(str, annotation)) + '\n')

# Process .json files
def process_json_file(json_file, image_dims, combined_file):
    index = os.path.splitext(os.path.basename(json_file))[0].split('_')[-1]
    image_filename = f'rgb_{index}.png'
    image_path = os.path.join(image_dir, image_filename)
    img_width, img_height = image_dims.get(image_filename, (1024, 768))

    with open(json_file) as f:
        data = json.load(f)

    yolo_annotations = []
    for bboxes in data.values():
        for bbox in bboxes:
            yolo_annotations.append(convert_bbox_to_yolo_format(bbox, img_width, img_height))

    with open(combined_file, 'a') as f:
        for annotation in yolo_annotations:
            f.write(' '.join(map(str, annotation)) + '\n')

def main():
    # Step 1: Get image dimensions
    image_dimensions = {}
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            image_dimensions[filename] = get_image_dimensions(image_path)

    # Ensure the combined file is empty or create it
    if os.path.exists(combined_yolo_file):
        os.remove(combined_yolo_file)
    
    # Step 2: Process .npy files
    for file in os.listdir(npy_dir):
        if file.lower().endswith('.npy'):
            npy_file = os.path.join(npy_dir, file)
            process_npy_file(npy_file, image_dimensions, combined_yolo_file)

    # Step 3: Process .json files
    for file in os.listdir(json_dir):
        if file.lower().endswith('.json'):
            json_file = os.path.join(json_dir, file)
            process_json_file(json_file, image_dimensions, combined_yolo_file)

if __name__ == '__main__':
    main()
