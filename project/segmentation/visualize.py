# visualize.py

import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_index(dataloader, index):
    # Fetch a batch from the dataloader
    images, masks = next(iter(dataloader))
    
    # Check if the index is within the batch size
    if index >= len(images):
        print(f"Index {index} is out of range for the current batch.")
        return

    # Select the image and mask at the given index
    image = images[index].permute(1, 2, 0).cpu().numpy()
    mask = masks[index].cpu().numpy()
    
    # Denormalize image
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)

    # Plot image and mask
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the image
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    # Display the mask with a colormap
    axes[1].imshow(mask, cmap='jet', vmin=0, vmax=np.max(mask))
    axes[1].set_title('Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
