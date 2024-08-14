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
    image = F.to_pil_image(image)
    boxes = target['boxes']
    labels = target['labels']
    
    print(f"Image index: {idx}")
    print(f"Number of bounding boxes: {len(boxes)}")
    #print(f"Labels: {labels}")
    
    # Convert PIL image to OpenCV format for better text handling
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box.int().tolist()
        # Draw rectangle
        cv2.rectangle(image_cv, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Red color in BGR
        # Draw label
        label_text = label
        cv2.putText(image_cv, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Green color

    # Convert back to RGB for displaying with matplotlib
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_cv)
    plt.axis('off')
    plt.title(f'Image index: {idx}')
    plt.show()

# Usage example
root_dir = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\train'

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add other transforms if needed
])

obj_det_dataset = ObjectDetectionDataset(root_dir, transform)

# Enter the index of the image you want to visualize
index_to_visualize = 33  # Change this index to any valid index you want to visualize
visualize_image_at_index(obj_det_dataset, index_to_visualize)
