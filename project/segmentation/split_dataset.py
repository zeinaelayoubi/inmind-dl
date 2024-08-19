import os
import shutil
from sklearn.model_selection import train_test_split

# Define your directories
image_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\images'
mask_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\masks'
label_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\labels'
train_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\train'
validate_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\val'

# Create directories for train and validate
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'masks'), exist_ok=True)
os.makedirs(os.path.join(validate_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(validate_dir, 'masks'), exist_ok=True)

# Collect all filenames
image_files = [f for f in os.listdir(image_dir) if f.startswith('rgb_') and f.endswith('.png')]
indices = [int(f.split('_')[1].split('.')[0]) for f in image_files]

# Split indices into train and validate
train_indices, validate_indices = train_test_split(indices, test_size=0.05, random_state=42)

def move_files(indices, src_image_dir, src_mask_dir, dst_image_dir, dst_mask_dir):
    for index in indices:
        # Ensure index is zero-padded to match filenames
        index_str = f"{index:04d}"
        
        # Construct filenames
        image_file = f"rgb_{index_str}.png"
        mask_file = f"semantic_segmentation_{index_str}.png"
        label_file = f"semantic_segmentation_labels_{index_str}.json"
        
        # Move image files
        src_image_path = os.path.join(src_image_dir, image_file)
        dst_image_path = os.path.join(dst_image_dir, image_file)
        if os.path.exists(src_image_path):
            shutil.move(src_image_path, dst_image_path)
        
        # Move mask files
        src_mask_path = os.path.join(src_mask_dir, mask_file)
        dst_mask_path = os.path.join(dst_mask_dir, mask_file)
        if os.path.exists(src_mask_path):
            shutil.move(src_mask_path, dst_mask_path)
        
        # Optionally move label files if needed
        src_label_path = os.path.join(label_dir, label_file)
        dst_label_path = os.path.join(dst_image_dir, label_file)
        if os.path.exists(src_label_path):
            shutil.move(src_label_path, dst_label_path)

# Move train and validate files
move_files(train_indices, image_dir, mask_dir, os.path.join(train_dir, 'images'), os.path.join(train_dir, 'masks'))
move_files(validate_indices, image_dir, mask_dir, os.path.join(validate_dir, 'images'), os.path.join(validate_dir, 'masks'))
