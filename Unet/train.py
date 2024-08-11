import torch
import os
import shutil
from dataset import CarvanaDataset
from model import UNET
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import save_checkpoint, load_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs
import torchvision.transforms as transforms
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

learning_rate = 1e-4
batch_size = 16
num_epochs = 1
num_workers = 4
pin_memory = True

# Directories
masks_dir = r"C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\train_masks"
images_dir = r"C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\train"
val_images_dir = r"C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\train_val"
val_masks_dir = r"C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\train_masks_val"

# Create validation and training directories
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_masks_dir, exist_ok=True)

# Split data into training and validation sets
def split_data(images_dir, masks_dir, val_images_dir, val_masks_dir, val_split=0.1):
    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    train_images, val_images = train_test_split(images, test_size=val_split, random_state=42)

    for img in val_images:
        # Move images
        shutil.move(os.path.join(images_dir, img), os.path.join(val_images_dir, img))
        
        # Move corresponding masks
        masks_file = img.replace('.jpg', '_mask.gif')
        shutil.move(os.path.join(masks_dir, masks_file), os.path.join(val_masks_dir, masks_file))

split_data(images_dir, masks_dir, val_images_dir, val_masks_dir)

if __name__ == '__main__':  # Added to prevent multiprocessing issues
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Load data
    train_loader, val_loader = get_loaders(
        train_dir=images_dir,
        train_maskdir=masks_dir,
        val_dir=val_images_dir,
        val_maskdir=val_masks_dir,
        batch_size=batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Initialize network
    model = UNET(in_channels=3, out_channels=1).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_loader, leave=True)

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.to(device).unsqueeze(1)

            # Forward pass
            predictions = model(data)
            loss = loss_fn(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        # Save checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # Check accuracy on validation set
        check_accuracy(val_loader, model, device=device)

        # Save some predictions to a folder
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=device)
