import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import json

class ObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, bboxes_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.bboxes_dir = bboxes_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # Collect all filenames
        self.rgb_files = sorted([f for f in os.listdir(images_dir) if f.startswith('rgb_') and f.endswith('.png')])
        self.bbox_files = sorted([f for f in os.listdir(bboxes_dir) if f.startswith('bounding_box_2d_tight_') and f.endswith('.npy')])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.startswith('bounding_box_2d_tight_labels_') and f.endswith('.json')])

        # Ensure that all directories have the same number of files
        assert len(self.rgb_files) == len(self.bbox_files) == len(self.label_files), "Mismatch in number of files across directories"

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Construct file paths
        img_path = os.path.join(self.images_dir, self.rgb_files[idx])
        bbox_path = os.path.join(self.bboxes_dir, self.bbox_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Load bounding boxes
        bboxes = np.load(bbox_path, allow_pickle=True)
        bboxes = np.array([list(item) for item in bboxes], dtype=np.float32)
        
        # Extract bounding box coordinates
        boxes = torch.tensor(bboxes[:, 1:5], dtype=torch.float32)  # x_min, y_min, x_max, y_max
        
        # Load labels
        with open(label_path, 'r') as f:
            labels_dict = json.load(f)
        
        # Map indices to class names
        class_names = {int(k): v["class"] for k, v in labels_dict.items()}
        
        # Convert indices to class names and match them with boxes
        labels = [class_names.get(int(bbox[0]), "unknown") for bbox in bboxes]
        
        # Convert labels to tensor of integers
        label_indices = torch.tensor([int(bbox[0]) for bbox in bboxes], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        target = {"boxes": boxes, "labels": label_indices}

        return image, target

# Example usage:
# Define directories
images_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\images'
bboxes_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\npy'
labels_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\json'

# Define transforms (optional)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Example transformation
    transforms.ToTensor()
])

# Create dataset
dataset = ObjectDetectionDataset(images_dir, bboxes_dir, labels_dir, transform=transform)
