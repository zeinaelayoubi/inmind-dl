import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
        self.labels = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.images[idx])
        label_path = os.path.join(self.data_dir, self.labels[idx])

        image = Image.open(image_path).convert("RGBA")
        with open(label_path, 'r') as f:
            label_map = json.load(f)

        # Convert RGBA image to class index mask
        image_array = np.array(image)
        mask = np.zeros(image_array.shape[:2], dtype=np.int64)

        for rgba_str, class_info in label_map.items():
            rgba_tuple = tuple(map(int, rgba_str.strip('()').split(',')))
            mask[np.all(image_array == rgba_tuple, axis=-1)] = int(class_info['class_id'])  # Assuming you have a 'class_id' in the JSON

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).long()  # Convert to PyTorch tensor

        return image, mask

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])

# Create dataset and dataloader
data_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset'

dataset = SegmentationDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
