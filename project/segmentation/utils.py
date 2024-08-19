import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
import matplotlib.pyplot as plt
from torchvision import transforms


# Define the color map for visualization
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

DEBUG = False  # Set to True to enable visualization

def pad_image(image, target_size):
    """ Pad the image to the target size. """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    
    if orig_width > orig_height:
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(target_size[1] * aspect_ratio)
    
    image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    new_image = Image.new('RGB', target_size, (0, 0, 0))  # Black padding
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_image.paste(image_resized, (paste_x, paste_y))
    
    return new_image

def pad_image_and_mask(image, mask, target_size):
    """ Pad both image and mask to the target size. """
    padded_image = pad_image(image, target_size)
    padded_mask = pad_image(mask.convert('L'), target_size)  # Convert mask to grayscale for padding
    return padded_image, padded_mask

# Define the transform pipeline
def get_transform(target_size):
    return transforms.Compose([
        transforms.Lambda(lambda img: pad_image(img, target_size)),  # Apply padding
        transforms.ToTensor()
    ])

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
    
def visualize_image_and_mask(image, mask, title=''):
    """
    Visualize image and mask side-by-side for debugging purposes.
    """
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze(0).cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f'{title} - Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f'{title} - Mask')
    plt.imshow(mask, cmap='jet')
    plt.axis('off')
    
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

