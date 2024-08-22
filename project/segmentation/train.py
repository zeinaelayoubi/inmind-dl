import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import SimpleSegNet
from utils import get_loaders, save_checkpoint, load_checkpoint, check_accuracy, save_predictions_as_imgs
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

DEBUG = True  # Set to True to enable visualization

# Define the transforms
train_transform = transforms.Compose([
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
])

def visualize_saved_predictions(folder='saved_images_epoch_1'):
    """
    Visualize saved predicted masks from the specified folder.
    """
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        return

    filenames = [f for f in os.listdir(folder) if f.endswith('.png')]
    filenames.sort()

    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(f'Prediction: {filename}')
        plt.axis('off')
        plt.show()

def main():
    # Configuration
    train_image_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\train\images'
    train_mask_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\train\masks'
    val_image_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\val\images'
    val_mask_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\val\masks'
    batch_size = 4
    num_workers = 2
    pin_memory = False
    num_classes = 11

    # DataLoader
    train_loader, val_loader = get_loaders(
        train_image_dir, train_mask_dir,
        val_image_dir, val_mask_dir,
        batch_size, train_transform, val_transform,
        num_workers, pin_memory
    )

    # Model
    device = torch.device('cpu')  # Force to use CPU
    model = SimpleSegNet(num_classes=num_classes).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter('runs/semantic_segmentation_experiment')

    # Initialize best validation accuracy
    best_val_accuracy = 0.0

    # Training Loop
    num_epochs = 4

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Train the model
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images, masks = batch['image'].to(device), batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Ensure outputs and targets are the correct shapes
            if outputs.dim() != 4 or masks.dim() != 4:
                raise ValueError(f"Outputs shape: {outputs.shape}, Masks shape: {masks.shape}")

            # Remove the extra channel dimension from masks if present
            if masks.shape[1] == 1:  # If masks have an extra channel dimension
                masks = masks.squeeze(1)  # Remove channel dimension

            # Ensure masks are in the right format
            if masks.dim() == 3:  # For mask: [batch_size, height, width]
                masks = masks.long()  # Ensure masks are long type for CrossEntropyLoss

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        writer.add_scalar('Loss/train', epoch_loss, epoch)

        # Check accuracy
        val_accuracy = check_accuracy(val_loader, model, device)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(model, optimizer, epoch, best_val_accuracy, 'best_model_weights.pt')

        # Save predictions for visualization
        if DEBUG:
            save_predictions_as_imgs(val_loader, model, folder='saved_images_epoch_{}'.format(epoch + 1), device=device)
            #visualize_saved_predictions(folder='saved_images_epoch_{}'.format(epoch + 1))

    writer.close()

if __name__ == "__main__":
    main()
