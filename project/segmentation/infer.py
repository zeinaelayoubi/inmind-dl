from PIL import Image
from torchvision import transforms
import numpy as np
import torch
from model import SimpleSegNet

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

def colorize_mask(mask, color_map):
    """Apply the color map to the mask."""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in color_map.items():
        color_mask[mask == class_idx] = color[:3]
    return color_mask

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleSegNet(num_classes=11).to(device)
checkpoint = torch.load('/content/your-repo/checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

def infer_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        output = model(image)
        pred_mask = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()

    return pred_mask

# Example usage
image_path = '/content/your-repo/path_to_your_image.jpg'
predicted_mask = infer_image(image_path)

# Apply color map and save the result
colorized_mask = colorize_mask(predicted_mask, COLOR_MAP)
colorized_mask_image = Image.fromarray(colorized_mask)
colorized_mask_image.save('/content/your-repo/predicted_mask.png')

# Visualize the result
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.imshow(colorized_mask)
plt.title('Predicted Mask')
plt.axis('off')
plt.show()
