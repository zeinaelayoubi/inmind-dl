import os
from torch.utils.data import Dataset
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms

class CarvanaDataset(Dataset):
    def __init__(self, masks_dir, images_dir, transform=None):
        self.masks_dir = masks_dir  # Updated argument name
        self.images_dir = images_dir  # Updated argument name
        self.transform = transform
        self.files = os.listdir(images_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, self.files[index])
        mask_path = os.path.join(self.masks_dir, self.files[index].replace('.png', '_mask.gif'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert mask to binary (assuming 255 indicates the object)
        mask = np.array(mask)
        mask[mask == 255] = 1
        
        if self.transform:
            image = self.transform(image)

        return image, mask
