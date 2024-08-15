import os
import shutil
from sklearn.model_selection import train_test_split

# Define the dataset directories
images_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\images'
bboxes_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\npy'
labels_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\json'
yolo_labels_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\yolo_labels'

# Define output directories for train and validation
object_detection_train_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\training_yolo\train'
object_detection_val_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\training_yolo\val'

# Create directories if they don't exist
for split_dir in [object_detection_train_dir, object_detection_val_dir]:
    os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'bboxes'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'yolo_labels'), exist_ok=True)

# Get list of all files
image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
bbox_files = [f for f in os.listdir(bboxes_dir) if f.endswith('.npy')]
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.json')]
yolo_files = [f for f in os.listdir(yolo_labels_dir) if f.endswith('.txt')]

# Ensure consistency between file lists
assert len(image_files) == len(bbox_files) == len(label_files) == len(yolo_files), "Mismatch between file counts"

# Create a mapping from file index to file name
def create_file_index_map(file_list):
    return {int(f.split('_')[-1].split('.')[0]): f for f in file_list}

image_index_map = create_file_index_map(image_files)
bbox_index_map = create_file_index_map(bbox_files)
label_index_map = create_file_index_map(label_files)
yolo_index_map = create_file_index_map(yolo_files)

# Sort the indices
sorted_indices = sorted(image_index_map.keys())

# Split indices into training and validation sets
train_indices, val_indices = train_test_split(sorted_indices, test_size=0.05, random_state=42)

def move_files_for_split(indices, split_dir):
    for idx in indices:
        # Define file names and paths
        img_file_name = image_index_map[idx]
        bbox_file_name = bbox_index_map[idx]
        lbl_file_name = label_index_map[idx]
        yolo_file_name = yolo_index_map[idx]
        
        src_paths = {
            'images': os.path.join(images_dir, img_file_name),
            'bboxes': os.path.join(bboxes_dir, bbox_file_name),
            'labels': os.path.join(labels_dir, lbl_file_name),
            'yolo_labels': os.path.join(yolo_labels_dir, yolo_file_name)
        }
        
        dst_paths = {
            'images': os.path.join(split_dir, 'images', img_file_name),
            'bboxes': os.path.join(split_dir, 'bboxes', bbox_file_name),
            'labels': os.path.join(split_dir, 'labels', lbl_file_name),
            'yolo_labels': os.path.join(split_dir, 'yolo_labels', yolo_file_name)
        }
        
        # Move files if they exist
        for key in src_paths:
            if os.path.exists(src_paths[key]):
                shutil.move(src_paths[key], dst_paths[key])

# Move files for training and validation splits
move_files_for_split(train_indices, object_detection_train_dir)
move_files_for_split(val_indices, object_detection_val_dir)

print("Datasets split into training and validation sets successfully.")
