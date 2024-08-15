

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
import cv2
from dataset import ObjectDetectionDataset
import torch
import numpy as np
from torchvision import transforms

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
        # Draw rectangle
        cv2.rectangle(image_cv, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Red color in BGR
        # Draw label
        label_text = str(label.item())  # Convert tensor to string
        cv2.putText(image_cv, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Green color

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
    transforms.RandomHorizontalFlip(p=0.1)
    # Add other transforms if needed
])

# Create dataset instance
obj_det_dataset = ObjectDetectionDataset(images_dir, bboxes_dir, labels_dir, transform=transform)

# Enter the index of the image you want to visualize
index_to_visualize = 653  # Change this index to any valid index you want to visualize
visualize_image_at_index(obj_det_dataset, index_to_visualize)
