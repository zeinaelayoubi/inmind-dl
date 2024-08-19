import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import HRNetV2Inspired
from utils import get_loaders, save_checkpoint, load_checkpoint, check_accuracy, save_predictions_as_imgs, visualize_image_and_mask
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

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

DEBUG = True  # Set to True to enable visualization

def pad_image(image, target_size):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    if orig_width > orig_height:
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(target_size[1] * aspect_ratio)

    image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    new_image = Image.new('RGB', target_size, (0, 0, 0))
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_image.paste(image_resized, (paste_x, paste_y))

    return new_image

def pad_image_train(image):
    return pad_image(image, (256, 256))

def pad_image_val(image):
    return pad_image(image, (256, 256))

# Define the transforms with padding
train_transform = transforms.Compose([
    transforms.Lambda(pad_image_train),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Lambda(pad_image_val),
    transforms.ToTensor(),
])

def visualize_saved_predictions(folder='saved_images_epoch_1'):
    """
    Visualize saved predicted masks from the specified folder.
    """
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        return

    # Get list of saved images
    filenames = [f for f in os.listdir(folder) if f.endswith('.png')]

    # Sort filenames if necessary
    filenames.sort()

    for filename in filenames:
        img_path = os.path.join(folder, filename)
        
        # Load the image
        img = Image.open(img_path)
        
        # Display the image
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
    batch_size = 2
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HRNetV2Inspired(in_channels=3, num_classes=num_classes).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Mixed Precision Scaler
    scaler = GradScaler()

    # Training Loop
    num_epochs = 1

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Create a tqdm progress bar for the training loop
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                masks = masks.squeeze(1).long()  # Remove the channel dimension

                # Print input image stats
                print(f"Input image stats - min: {images.min().item()}, max: {images.max().item()}")
                
                # Visualize input image and mask
                if DEBUG and batch_idx == 0:
                    visualize_image_and_mask(images[0].cpu(), masks[0].cpu(), title='Input Image and Mask')

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass with autocast
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                # Print output stats
                print(f"Output stats - min: {outputs.min().item()}, max: {outputs.max().item()}")

                # Visualize output mask
                if DEBUG and batch_idx == 0:
                    pred_mask = torch.argmax(outputs[0], dim=0).cpu()
                    visualize_image_and_mask(images[0].cpu(), pred_mask, title='Predicted Mask')

                # Backward pass and optimize with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)

                # Clear CUDA cache
                torch.cuda.empty_cache()

                # Update progress bar
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validation
        print("Evaluating on validation set...")
        check_accuracy(val_loader, model, device=device)

        # Save checkpoint
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename='checkpoint.pth.tar')

        # Save predictions
        prediction_folder = f'saved_images_epoch_{epoch + 1}'
        save_predictions_as_imgs(val_loader, model, folder=prediction_folder, device=device)

        # Visualize saved predictions
        visualize_saved_predictions(prediction_folder)

if __name__ == "__main__":
    main()
