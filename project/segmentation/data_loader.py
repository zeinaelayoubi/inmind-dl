# data_loader.py

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class CustomSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, json_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.json_dir = json_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        self.json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
        
        # Load label map from JSON
        self.label_map = {}
        for json_file in self.json_files:
            with open(os.path.join(self.json_dir, json_file), 'r') as f:
                labels = json.load(f)
                for rgba_str, info in labels.items():
                    rgba = tuple(map(int, rgba_str.strip('()').split(',')))
                    self.label_map[rgba] = info['class']
        
        # Map classes to IDs
        class_names = sorted(set(self.label_map.values()))
        self.class_map = {cls: idx for idx, cls in enumerate(class_names)}
        self.rgba_to_id = {rgba: self.class_map[cls] for rgba, cls in self.label_map.items()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGBA")

        # Convert RGBA to class IDs
        mask_ids = np.zeros((mask.height, mask.width), dtype=np.int64)
        for rgba, class_id in self.rgba_to_id.items():
            mask_ids[(np.array(mask) == np.array(rgba)).all(axis=-1)] = class_id

        mask_ids = Image.fromarray(mask_ids)

        if self.transform:
            augmented = self.transform(image=image, mask=mask_ids)
            image = augmented['image']
            mask_ids = augmented['mask']

        return image, mask_ids

# Define your augmentations using torchvision.transforms
class CustomTransforms:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image, mask):
        image = self.transforms(image)
        mask = transforms.ToTensor()(mask)  # Convert mask to tensor
        return {'image': image, 'mask': mask}

def get_dataloader(image_dir, mask_dir, json_dir, batch_size=4, num_workers=2):
    transform = CustomTransforms()
    dataset = CustomSegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, json_dir=json_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print(f"Number of images: {len(dataset.images)}")
    print(f"Number of masks: {len(dataset.masks)}")

    return dataloader
