import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import HRNetV2Inspired

# Define the color map
COLOR_MAP = {
    (0, 0, 0, 0): 0,          # BACKGROUND
    (25, 255, 82, 255): 1,    # iwhub
    (25, 82, 255, 255): 2,    # dolly
    (255, 25, 197, 255): 3,   # pallet
    (140, 25, 255, 255): 4,   # crate
    (140, 255, 25, 255): 5,   # rack
    (255, 111, 25, 255): 6,   # railing
    (0, 0, 0, 255): 7,        # UNLABELLED
    (226, 255, 25, 255): 8,   # floor
    (255, 197, 25, 255): 9,   # forklift
    (54, 255, 25, 255): 10    # stillage
}

def load_model(checkpoint_path, device='cpu'):
    model = HRNetV2Inspired(in_channels=3, num_classes=len(COLOR_MAP))  # Adjust num_classes
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'])
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def postprocess_output(output, color_map):
    output = torch.argmax(output.squeeze(0), dim=0).byte().cpu().numpy()
    
    # Create an image with the same size as the output
    color_output = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    
    # Reverse the color map for easy lookup
    reverse_color_map = {v: k for k, v in color_map.items()}
    
    # Map class indices to colors
    for class_index, color in reverse_color_map.items():
        color_output[output == class_index] = color[:3]  # Use RGB only
    
    return color_output

def main():
    # Configuration
    image_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\train\images\rgb_0956.png'  # Path to the input image
    checkpoint_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\segmentation\checkpoint.pth.tar'  # Path to the saved model checkpoint
    output_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\segmentation'  # Path to save the segmented output

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    model = load_model(checkpoint_path, device=device)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load and preprocess image
    image = preprocess_image(image_path, transform)
    image = image.to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)

    # Post-process output
    segmented_image = postprocess_output(output, COLOR_MAP)

    # Save or display output
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    main()
