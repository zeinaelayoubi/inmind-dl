from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from torchvision import transforms
from torchvision.transforms import functional as F

# Define the directory containing the mask images
mask_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\train\masks'

# Specify a mask file (replace 'semantic_segmentation_0001.png' with an actual file name)
mask_file = 'semantic_segmentation_0001.png'
mask_path = os.path.join(mask_dir, mask_file)

# Open the mask image file
mask_image = Image.open(mask_path).convert('RGBA')
print(f"Mask mode: {mask_image.mode}")  # Should output 'RGBA'
print(f"Mask size: {mask_image.size}")  # Print size to verify it's what you expect

# Define the transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10)
])

def apply_transforms(image, transform):
    # Apply the transformation to the image
    return transform(image)

# Apply transformations to the mask
mask_image_transformed = apply_transforms(mask_image, transform)

# Convert transformed mask to numpy array
mask_array_transformed = np.array(mask_image_transformed)

# Plot the original and transformed mask images
fig, axes = plt.subplots(1, 2, figsize=(12, 7))  # Two subplots: one for original mask, one for transformed mask
axes[0].imshow(np.array(mask_image))
axes[0].set_title('Original Mask')
axes[0].axis('off')

axes[1].imshow(mask_array_transformed)
axes[1].set_title('Transformed Mask')
axes[1].axis('off')

plt.tight_layout()
plt.show()
