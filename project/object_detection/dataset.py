import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import json

class ObjectDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.rgb_files = sorted([f for f in os.listdir(root_dir) if f.startswith('rgb_') and f.endswith('.png')])
        self.bbox_files = sorted([f for f in os.listdir(root_dir) if f.startswith('bounding_box_2d_tight_') and f.endswith('.npy')])
        self.label_files = sorted([f for f in os.listdir(root_dir) if f.startswith('bounding_box_2d_tight_labels_') and f.endswith('.json')])

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.rgb_files[idx])
        bbox_path = os.path.join(self.root_dir, self.bbox_files[idx])
        label_path = os.path.join(self.root_dir, self.label_files[idx])
        
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
        labels = [class_names[int(bbox[0])] for bbox in bboxes]
        
        # Convert labels to tensor
        labels = labels  # Keep labels as strings for display purposes

        if self.transform:
            image = self.transform(image)

        target = {"boxes": boxes, "labels": labels}

        return image, target
