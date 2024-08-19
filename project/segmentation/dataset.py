import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Collect and sort filenames
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

        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGBA')  # Load as RGBA

        # Convert RGBA mask to single-channel mask
        mask = np.array(mask)
        class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        
        for rgba, class_index in COLOR_MAP.items():
            mask_class = np.all(mask == rgba, axis=-1)
            class_mask[mask_class] = class_index
        
        mask = Image.fromarray(class_mask)

        # Transform images and masks if provided
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)  # Apply target_transform to mask

        return {'image': image, 'mask': mask}


# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # Convert images to tensor
    ])

    # Define a target transform if needed
    target_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # Convert masks to tensor if needed
    ])

    dataset = SegmentationDataset(
        image_dir=r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\train\images',
        mask_dir=r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\train\masks',
        transform=transform,
        target_transform=target_transform
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Just a single print statement after loading the dataset
    print("Dataset loaded successfully")

    # Optional: To check that data is loaded correctly without printing everything
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 0:  # Just check the first batch
            images = batch['image']
            masks = batch['mask']
            print(f"First batch - Image shape: {images.shape}, Mask shape: {masks.shape}")
            break
