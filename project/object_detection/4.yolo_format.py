import os
import numpy as np
from PIL import Image

def convert_bbox_to_yolo_format(image_width, image_height, bbox):
    bbox_semantic_id, x_min, y_min, x_max, y_max, _ = bbox
    
    # Calculate center, width, and height in YOLO format
    x_center = (x_min + x_max) / 2.0 / image_width
    y_center = (y_min + y_max) / 2.0 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    
    return f"{bbox_semantic_id} {x_center} {y_center} {width} {height}"

def process_files(npy_folder, json_folder, image_folder, labels_folder):
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    
    for idx in range(1000):  # Assuming 0000 to 0999 files
        npy_file = os.path.join(npy_folder, f"bounding_box_2d_tight_{idx:04d}.npy")
        json_file = os.path.join(json_folder, f"bounding_box_2d_tight_labels_{idx:04d}.json")
        image_file = os.path.join(image_folder, f"rgb_{idx:04d}.png")
        
        if not os.path.isfile(npy_file) or not os.path.isfile(json_file) or not os.path.isfile(image_file):
            print(f"Missing file(s) for index {idx}")
            continue
        
        # Load bounding boxes
        bboxes = np.load(npy_file)
        
        # Load image to get dimensions
        with Image.open(image_file) as img:
            image_width, image_height = img.size
        
        # Create YOLO format annotations
        yolo_annotations = []
        for bbox in bboxes:
            yolo_annotations.append(convert_bbox_to_yolo_format(image_width, image_height, bbox))
        
        # Write annotations to text file
        output_txt_file = os.path.join(labels_folder, f"{idx:04d}.txt")
        with open(output_txt_file, 'w') as f:
            f.write("\n".join(yolo_annotations))
        print(f"Processed {output_txt_file}")

# Directories
npy_folder = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\npy'
json_folder = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\json'
image_folder = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\images'
labels_folder = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\yolo_labels'

# Process the files
process_files(npy_folder, json_folder, image_folder, labels_folder)
