import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.colors as mcolors

# Define the RGBA to class index mapping
COLOR_MAP = {
    0: (0, 0, 0, 0),          # BACKGROUND
    1: (25, 255, 82, 255),    # iwhub
    2: (25, 82, 255, 255),    # dolly
    3: (255, 25, 197, 255),   # pallet
    4: (140, 25, 255, 255),   # crate
    5: (140, 255, 25, 255),   # rack
    6: (255, 111, 25, 255),   # railing
    7: (0, 0, 0, 255),        # UNLABELLED
    8: (226, 255, 25, 255),   # floor
    9: (255, 197, 25, 255),   # forklift
    10: (54, 255, 25, 255)    # stillage
}

def create_custom_colormap(color_map):
    colors = [color_map[i] for i in sorted(color_map.keys())]
    colors = [(r/255, g/255, b/255) for (r, g, b, a) in colors]
    cmap = mcolors.ListedColormap(colors)
    return cmap

custom_cmap = create_custom_colormap(COLOR_MAP)

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') and f.startswith('rgb_')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png') and f.startswith('semantic_segmentation_')])

        if len(self.image_files) != len(self.mask_files):
            raise ValueError("Mismatch between the number of images and masks.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        mask_file = self.mask_files[idx]

        image_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGBA')

        mask = np.array(mask)
        class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        for class_index, rgba in COLOR_MAP.items():
            mask_class = np.all(mask[:, :, :4] == rgba, axis=-1)
            class_mask[mask_class] = class_index

        mask = Image.fromarray(class_mask, mode='L')

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)

        return {'image': image, 'mask': mask}
