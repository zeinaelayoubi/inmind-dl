import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.colors as mcolors
from dataset import SegmentationDataset

COLOR_MAP = {
    0: (0, 0, 0, 0),
    1: (25, 255, 82, 255),
    2: (25, 82, 255, 255),
    3: (255, 25, 197, 255),
    4: (140, 25, 255, 255),
    5: (140, 255, 25, 255),
    6: (255, 111, 25, 255),
    7: (0, 0, 0, 255),
    8: (226, 255, 25, 255),
    9: (255, 197, 25, 255),
    10: (54, 255, 25, 255)
}

def create_custom_colormap(color_map):
    colors = [color_map[i] for i in sorted(color_map.keys())]
    colors = [(r/255, g/255, b/255) for (r, g, b, a) in colors]
    cmap = mcolors.ListedColormap(colors)
    return cmap

custom_cmap = create_custom_colormap(COLOR_MAP)

def colorize_segmentation(pred_mask_np, color_map):
    color_mask = np.zeros((pred_mask_np.shape[0], pred_mask_np.shape[1], 4), dtype=np.uint8)
    for class_index, color in color_map.items():
        color_mask[pred_mask_np == class_index] = color
    return color_mask

def get_loaders(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir,
                batch_size, train_transform, val_transform,
                num_workers=4, pin_memory=True):
    train_dataset = SegmentationDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
        target_transform=train_transform
    )
    val_dataset = SegmentationDataset(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
        target_transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader

def save_checkpoint(model, optimizer, epoch, best_accuracy, filename='checkpoint.pth'):
    print(f"Saving checkpoint to {filename}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy
    }, filename)

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    print(f"Loaded checkpoint from {filename}")
    return model, optimizer, epoch, best_accuracy

def save_predictions_as_imgs(loader, model, folder='saved_images', device='cpu'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for j in range(preds.size(0)):
                pred = preds[j].cpu().numpy()
                colored_mask = colorize_segmentation(pred, COLOR_MAP)
                colored_mask_pil = Image.fromarray(colored_mask, mode='RGBA')
                colored_mask_pil.save(os.path.join(folder, f'pred_{i * loader.batch_size + j}.png'))
                
                if i < 1:  # Limit number of images for visualization
                    image_pil = transforms.ToPILImage()(images[j].cpu())
                    image_pil.save(os.path.join(folder, f'image_{i * loader.batch_size + j}.png'))

    print(f"Saved predictions to {folder}")
    
def check_accuracy(loader, model, device='cpu'):
    model.eval()
    num_correct = 0
    num_pixels = 0
    with torch.no_grad():
        for batch in loader:
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            num_correct += torch.sum(preds == masks).item()
            num_pixels += torch.numel(masks)
    accuracy = num_correct / num_pixels
    print("accuracy:",accuracy)
    return accuracy