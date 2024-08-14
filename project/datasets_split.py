import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Define the dataset directory
dataset_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset'

# Define output directories for train and validation
object_detection_train_dir = os.path.join(dataset_dir, 'object_detection', 'train')
object_detection_val_dir = os.path.join(dataset_dir, 'object_detection', 'val')
segmentation_train_dir = os.path.join(dataset_dir, 'semantic_segmentation', 'train')
segmentation_val_dir = os.path.join(dataset_dir, 'semantic_segmentation', 'val')

# Create directories if they don't exist
os.makedirs(object_detection_train_dir, exist_ok=True)
os.makedirs(object_detection_val_dir, exist_ok=True)
os.makedirs(segmentation_train_dir, exist_ok=True)
os.makedirs(segmentation_val_dir, exist_ok=True)

# Get list of all files
all_files = os.listdir(dataset_dir)

# Separate files into object detection and semantic segmentation
def split_and_move_files(file_list, train_dir, val_dir, split_ratio=0.05):
    # Create a mapping from file index to file name
    file_index_map = {int(f.split('_')[-1].split('.')[0]): f for f in file_list}
    
    # Sort the file indices
    sorted_indices = sorted(file_index_map.keys())
    
    # Split indices into training and validation sets
    train_indices, val_indices = train_test_split(sorted_indices, test_size=split_ratio, random_state=42)
    
    for idx in train_indices:
        file_name = file_index_map[idx]
        shutil.move(os.path.join(dataset_dir, file_name), os.path.join(train_dir, file_name))
    for idx in val_indices:
        file_name = file_index_map[idx]
        shutil.move(os.path.join(dataset_dir, file_name), os.path.join(val_dir, file_name))

# Object Detection Files
bbox_files = [f for f in all_files if f.startswith('bounding_box_2d_tight_') and f.endswith('.npy')]
image_files = [f for f in all_files if f.startswith('rgb_') and f.endswith('.png')]
label_files = [f for f in all_files if f.startswith('bounding_box_2d_tight_labels_') and f.endswith('.json')]
prim_paths_files = [f for f in all_files if f.startswith('bounding_box_2d_tight_prim_paths_') and f.endswith('.json')]

# Ensure consistency between file lists
assert len(bbox_files) == len(image_files) == len(label_files) == len(prim_paths_files), "Mismatch between file counts"

# Split and move object detection files
split_and_move_files(bbox_files, object_detection_train_dir, object_detection_val_dir)
split_and_move_files(image_files, object_detection_train_dir, object_detection_val_dir)
split_and_move_files(label_files, object_detection_train_dir, object_detection_val_dir)
split_and_move_files(prim_paths_files, object_detection_train_dir, object_detection_val_dir)

# Semantic Segmentation Files
segmentation_image_files = [f for f in all_files if f.startswith('semantic_segmentation_') and f.endswith('.png')]
segmentation_label_files = [f for f in all_files if f.startswith('semantic_segmentation_labels_') and f.endswith('.json')]

# Ensure consistency between file lists
assert len(segmentation_image_files) == len(segmentation_label_files), "Mismatch between image and label counts"

# Create a mapping from file index to file name for semantic segmentation
segmentation_file_index_map = {int(f.split('_')[-1].split('.')[0]): f for f in segmentation_image_files}
segmentation_label_index_map = {int(f.split('_')[-1].split('.')[0]): f for f in segmentation_label_files}

# Sort the indices
segmentation_indices = sorted(segmentation_file_index_map.keys())

# Split indices into training and validation sets
segmentation_train_indices, segmentation_val_indices = train_test_split(segmentation_indices, test_size=0.05, random_state=42)

# Move files for semantic segmentation
for idx in segmentation_train_indices:
    img_file_name = segmentation_file_index_map[idx]
    lbl_file_name = segmentation_label_index_map[idx]
    shutil.move(os.path.join(dataset_dir, img_file_name), os.path.join(segmentation_train_dir, img_file_name))
    shutil.move(os.path.join(dataset_dir, lbl_file_name), os.path.join(segmentation_train_dir, lbl_file_name))

for idx in segmentation_val_indices:
    img_file_name = segmentation_file_index_map[idx]
    lbl_file_name = segmentation_label_index_map[idx]
    shutil.move(os.path.join(dataset_dir, img_file_name), os.path.join(segmentation_val_dir, img_file_name))
    shutil.move(os.path.join(dataset_dir, lbl_file_name), os.path.join(segmentation_val_dir, lbl_file_name))

print("Datasets split into training and validation sets successfully.")
