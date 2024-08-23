import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import time  # Import the time module
from model import SimpleSegNet
from utils import colorize_segmentation, COLOR_MAP  # Import colorize_segmentation and COLOR_MAP

def load_model_from_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from a checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        # Extract model state_dict from checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If no 'model_state_dict' key, assume the checkpoint is directly the model's state_dict
        model.load_state_dict(checkpoint)
    
    return model

def inference(image_path, model_weights_path):
    # Load the model
    device = 'cpu'
    model = SimpleSegNet(num_classes=11).to(device)
    model = load_model_from_checkpoint(model, model_weights_path, device)
    model.eval()

    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize([720, 1280]),  # Ensure the image size matches the model's input size
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Perform inference and time it
    start_time = time.time()  # Start timing
    with torch.no_grad():
        output = model(image)
        
        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(output, dim=1)
        
        # Get the prediction by taking the class with the highest probability
        pred_mask = torch.argmax(probabilities[0], dim=0).cpu()
    end_time = time.time()  # End timing
    
    # Calculate the elapsed time
    inference_time = end_time - start_time
    print(f'Inference time: {inference_time:.4f} seconds')

    # Debug: Print statistics of the predicted mask
    print(f'Predicted mask shape: {pred_mask.shape}')
    print(f'Predicted mask unique values: {np.unique(pred_mask.numpy())}')

    # Convert the mask to a numpy array
    pred_mask_np = pred_mask.numpy()

    # Apply color map to the predicted mask
    colored_mask = colorize_segmentation(pred_mask_np, COLOR_MAP)

    # Convert the colored mask to a PIL image
    colored_mask_pil = Image.fromarray(colored_mask, mode='RGBA')

    # Save or display the predicted mask
    colored_mask_pil.save('inferred_colored_mask.png')
    colored_mask_pil.show()

# Example usage
image_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\train\temp_images\rgb_0018.png'
model_weights_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\segmentation\best_model_weights.pt'
inference(image_path, model_weights_path)
