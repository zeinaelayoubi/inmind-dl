import sys
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Path to YOLOv5 repo and model
yolov5_repo = Path('yolov5')  # Adjust this path if necessary
model_path = yolov5_repo / 'runs' / 'train' / 'exp3' / 'weights' / 'best.pt'

# Add YOLOv5 repo to Python path
sys.path.append(str(yolov5_repo))

# Load YOLOv5 model using torch.hub
model = torch.hub.load(
    'ultralytics/yolov5',  # Repository
    'custom',              # Load a custom model
    path=str(model_path),  # Path to custom weights
    source='local',        # Load from local file system
    force_reload=False     # Optional: Set to True to force reload the model
)

model.eval()

# Function to preprocess image
def preprocess_image(image_path, img_size=416):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_size, img_size))
    image_np = np.array(image).astype(np.float32)
    image_np = np.transpose(image_np, (2, 0, 1))  # (H, W, C) to (C, H, W)
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
    image_np /= 255.0  # Normalize to [0, 1]
    return torch.from_numpy(image_np).float()

# Function to scale coordinates
def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].round()
    return coords

# Function to draw bounding boxes and labels
def draw_boxes(image_np, boxes, class_ids, class_names):
    for box, cls_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # Green color for bounding box
        label = f'{class_names[int(cls_id)]}'
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Path to your image
image_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\yolov5_format\val\images\rgb_0059.png'
img_size = 416

# Load and preprocess the image
input_tensor = preprocess_image(image_path, img_size=img_size).to('cuda' if torch.cuda.is_available() else 'cpu')

# Perform inference
with torch.no_grad():
    outputs = model(input_tensor)[0]  # Extract detections

# Post-process outputs
objectness_threshold = 0.3
class_names = model.names  # Get class names
predictions = outputs[outputs[:, 4] > objectness_threshold]
boxes = predictions[:, :4]
class_ids = predictions[:, 5]  # Get class IDs
boxes = scale_coords((img_size, img_size), boxes, Image.open(image_path).size)

# Draw bounding boxes and labels
image_np = np.array(Image.open(image_path))
draw_boxes(image_np, boxes, class_ids, class_names)

# Display image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
