import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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

# Function to create a custom colormap from the COLOR_MAP
def create_custom_colormap(color_map):
    """Create a custom colormap from the given color map."""
    colors = [color_map[i] for i in sorted(color_map.keys())]
    colors = [(r/255, g/255, b/255) for (r, g, b, a) in colors]
    cmap = mcolors.ListedColormap(colors)
    return cmap

# Create a custom colormap
custom_cmap = create_custom_colormap(COLOR_MAP)

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
        
        for class_index, rgba in COLOR_MAP.items():
            mask_class = np.all(mask[:, :, :4] == rgba, axis=-1)
            class_mask[mask_class] = class_index
        
        mask = Image.fromarray(class_mask, mode='L')

        # Transform images and masks if provided
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)  # Apply target_transform to mask

        return {'image': image, 'mask': mask}

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        #transforms.Resize((256, 256)),  # Resize images
        transforms.ToTensor(),  # Convert images to tensor
    ])

    target_transform = transforms.Compose([
        #transforms.Resize((256, 256)),  # Resize masks
        transforms.ToTensor(),  # Convert masks to tensor without normalization
        transforms.Lambda(lambda x: x.long())  # Ensure mask values are integers (class indices)
    ])

    dataset = SegmentationDataset(
        image_dir=r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\train\images',
        mask_dir=r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\train\masks',
        transform=transform,
        target_transform=target_transform
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Index of the sample to visualize
    sample_index = 24  # Change this index to view different samples

    if sample_index < len(dataset):
        sample = dataset[sample_index]
        image = sample['image']
        mask = sample['mask']

        # Convert tensors to PIL Images for visualization
        image_pil = transforms.ToPILImage()(image.cpu())
        mask_pil = transforms.ToPILImage()(mask.cpu().to(dtype=torch.uint8))  # Convert mask to uint8
        
        # Convert mask to a numpy array for visualization
        mask_np = np.array(mask_pil)

        # Print mask information
        print(f"Mask unique values: {np.unique(mask_np)}")

        # Load and verify the mask image directly
        mask_image = Image.open(os.path.join(dataset.mask_dir, dataset.mask_files[sample_index]))
        mask_array = np.array(mask_image)

        # Print unique RGBA values
        unique_rgba = np.unique(mask_array.reshape(-1, mask_array.shape[2]), axis=0)
        print("Unique RGBA values in mask image:")
        for rgba in unique_rgba:
            print(rgba)

        # Plot the image and mask with the custom colormap
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))  # Added an extra subplot for direct mask view
        axes[0].imshow(image_pil)
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask_np, cmap=custom_cmap)  # Use custom colormap for mask
        axes[1].set_title('Mask')
        axes[1].axis('off')
        
        # Set aspect ratio to 'equal' to avoid squeezing
        for ax in axes:
            ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()
    else:
        print("Sample index is out of range.")
