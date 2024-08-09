import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import shutil
from sklearn.model_selection import train_test_split

from dataset import CustomImageDataset
from model import UNET
from utils import save_checkpoint, load_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs
import tqdm 

device = "cuda" if torch.cuda.is_available() else "cpu"


learning_rate = 1e-4
batch_size = 16
num_epochs = 25
num_workers = 4
pin_memory = True


mask_dir = "C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\train_masks"
images_dir = "C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\train"
val_images_dir = "C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\train_val"
val_mask_dir = "C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\train_masks_val"

#  train and validation sets
def split_data(images_dir, mask_dir, val_images_dir, val_mask_dir, val_split=0.1):
    images = os.listdir(images_dir)
    train_images, val_images = train_test_split(images, test_size=val_split, random_state=42)

    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    for img in val_images:
        shutil.move(os.path.join(images_dir, img), val_images_dir)
        shutil.move(os.path.join(mask_dir, img.replace('.png', '_mask.gif')), val_mask_dir)

    return train_images, val_images


split_data(images_dir, mask_dir, val_images_dir, val_mask_dir)

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
    images_dir, mask_dir, val_images_dir, val_mask_dir, batch_size, train_transform, val_transform, num_workers, pin_memory
)

# Initialize network
model = UNET(in_channels=3, out_channels=1).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    loop = tqdm(train_loader, leave=True)
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device).unsqueeze(1)

        # Forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # Backward
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

    # Save some examples to a folder
    save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=device)
