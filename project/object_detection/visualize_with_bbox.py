import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
import cv2
from dataset import ObjectDetectionDataset
import torch
import numpy as np
from torchvision import transforms

# Define the label mapping
label_mapping = {
    0: 'forklift',
    1: 'rack',
    2: 'crate',
    3: 'floor',
    4: 'railing',
    5: 'pallet',
    6: 'stillage',
    7: 'iwhub',
    8: 'dolly'
}

# Define colors for each label
color_mapping = {
    0: (0, 255, 255),  # Yellow
    1: (255, 0, 255),  # Magenta
    2: (255, 255, 0),  # Cyan
    3: (0, 255, 0),    # Green
    4: (255, 0, 0),    # Red
    5: (0, 0, 255),    # Blue
    6: (255, 128, 0),  # Orange
    7: (128, 0, 128),  # Purple
    8: (0, 128, 128)   # Teal
}

def visualize_image_at_index(dataset, idx):
    # Ensure index is valid
    if idx < 0 or idx >= len(dataset):
        raise IndexError(f"Index {idx} out of range. Dataset contains {len(dataset)} images.")
    
    # Get image and target at the specified index
    image, target = dataset[idx]
    boxes = target['boxes']
    labels = target['labels']

    # Convert image from Tensor to PIL
    image = F.to_pil_image(image)
    
    print(f"Image index: {idx}")
    print(f"Number of bounding boxes: {len(boxes)}")
    
    # Convert PIL image to OpenCV format for better text handling
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box.int().tolist()
        label_index = label.item()  # Convert tensor to int
        color = color_mapping.get(label_index, (0, 0, 0))  # Default to black if label not found
        label_text = label_mapping.get(label_index, 'Unknown')  # Get label name from mapping
        
        # Draw rectangle with the specific color
        cv2.rectangle(image_cv, (xmin, ymin), (xmax, ymax), color, 2)
        # Draw label text with the same color
        cv2.putText(image_cv, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Convert back to RGB for displaying with matplotlib
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_cv)
    plt.axis('off')
    plt.title(f'Image index: {idx}')
    plt.show()

# Define directories
images_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\images'
bboxes_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\npy'
labels_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\json'

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image to Tensor
    #transforms.RandomHorizontalFlip(p=0.1)
    # Add other transforms if needed
])

# Create dataset instance
obj_det_dataset = ObjectDetectionDataset(images_dir, bboxes_dir, labels_dir, transform=transform)

# Enter the index of the image you want to visualize
index_to_visualize = 543  # Change this index to any valid index you want to visualize
visualize_image_at_index(obj_det_dataset, index_to_visualize)
