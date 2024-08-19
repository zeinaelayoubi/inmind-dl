import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from dataset import SegmentationDataset, COLOR_MAP
import matplotlib.pyplot as plt
from torchvision import transforms


def get_loaders(
    train_image_dir,
    train_mask_dir,
    val_image_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True
):
    """
    Create DataLoader instances for training and validation datasets.
    """
    train_dataset = SegmentationDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
        target_transform=train_transform  # Apply same transform to masks
    )

    val_dataset = SegmentationDataset(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
        target_transform=val_transform  # Apply same transform to masks
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    """Save the model checkpoint."""
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """Load the model checkpoint."""
    model.load_state_dict(checkpoint['state_dict'])
    # Uncomment the following if you are using an optimizer
    # optimizer.load_state_dict(checkpoint['optimizer'])

def check_accuracy(loader, model, device='cuda'):
    """
    Check the accuracy of the model on the given data loader.
    """
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device=device)
            masks = batch['mask'].to(device=device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            num_correct += (preds == masks).sum().item()
            num_samples += torch.numel(preds)

    accuracy = num_correct / num_samples
    print(f'Accuracy: {accuracy:.4f}') 
    
def visualize_image_and_mask(dataset, index, title=''):
    """
    Visualize image and mask from the dataset at a specific index side-by-side for debugging purposes.
    """
    if index >= len(dataset):
        print("Index is out of range.")
        return

    # Get sample from dataset
    sample = dataset[index]
    image = sample['image']
    mask = sample['mask']

    # Convert tensors to PIL Images for visualization
    image_pil = transforms.ToPILImage()(image.cpu())
    mask_pil = transforms.ToPILImage()(mask.cpu().to(dtype=torch.uint8))  # Convert mask to uint8

    # Convert mask to a numpy array for visualization
    mask_np = np.array(mask_pil)

    # Print mask information
    print(f"Mask unique values: {np.unique(mask_np)}")

    mask_image = Image.open(os.path.join(dataset.mask_dir, dataset.mask_files[index]))
    mask_array = np.array(mask_image)
        #print("mask array",mask_array)
        # Print unique RGBA values
    unique_rgba = np.unique(mask_array.reshape(-1, mask_array.shape[2]), axis=0)
    print("Unique RGBA values in mask image:")
    for rgba in unique_rgba:
        print(rgba)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))  # Added an extra subplot for direct mask view
    axes[0].imshow(image_pil)
    axes[0].set_title('Image')
    axes[0].axis('off')
        
        
    axes[1].imshow(mask_array)  # Directly show the original mask with RGBA
    axes[1].set_title('Original Mask')
    axes[1].axis('off')
        
        # Set aspect ratio to 'equal' to avoid squeezing
    for ax in axes:
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

def colorize_segmentation(segmentation, color_map):
    """
    Colorize a segmentation mask according to the provided color map.
    """
    # Create an empty array to hold the colored segmentation
    colored_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 4), dtype=np.uint8)
    
    # Iterate over the color map
    for index, color in color_map.items():
        colored_segmentation[segmentation == index] = color

    return colored_segmentation

def save_predictions_as_imgs(loader, model, folder='saved_images', device='cuda'):
    """
    Save the model predictions as images.
    """
    model.eval()
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch['image'].to(device)
            filenames = batch.get('filename', [f'{batch_idx * len(images) + i}.png' for i in range(len(images))])
            outputs = model(images)
            
            if outputs.dim() == 4:  # Shape: [batch_size, num_classes, height, width]
                outputs = torch.argmax(outputs, dim=1)  # Convert logits to class indices
            elif outputs.dim() == 3:  # Shape: [batch_size, height, width] - already class indices
                outputs = outputs.unsqueeze(1)  # Add channel dimension
            else:
                raise ValueError(f"Unexpected output shape: {outputs.shape}")

            for i, filename in enumerate(filenames):
                output = outputs[i].cpu().numpy()  # Get the class index map for the image
                if output.ndim == 2:  # Shape: [height, width]
                    output_img = colorize_segmentation(output, COLOR_MAP)

                    # Save the output image
                    output_img = Image.fromarray(output_img, mode='RGBA')
                    output_img.save(os.path.join(folder, f'pred_{filename}'))
                else:
                    raise ValueError(f"Unexpected output image shape: {output.shape}")
