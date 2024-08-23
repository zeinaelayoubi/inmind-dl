import onnxruntime as ort
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import cv2
import torch

# Function to scale coordinates from the resized image to the original image
def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].round()
    return coords

# Function to read YOLO format labels from a file
def read_yolo_labels(label_path):
    with open(label_path, 'r') as f:
        labels = f.readlines()
    labels = [list(map(float, line.strip().split())) for line in labels]
    return np.array(labels)

# Function to calculate IoU between two bounding boxes
def calculate_iou(box1, box2):
    # Calculate the intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y1_inter - y2_inter + 1)
    
    # Calculate the areas of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    # Calculate the union area
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area
    return iou

# Class names and corresponding colors for bounding boxes
class_names = [
    "forklift", "rack", "crate", "floor", "railing", 
    "pallet", "stillage", "iwhub", "dolly"
]
class_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0), 
    (0, 128, 128)
]

# Load the ONNX model
ort_session = ort.InferenceSession(r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\object_detection\yolov5\runs\train\exp3\weights\best.onnx')

# Load and preprocess the image
image_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\yolov5_format\train\images\rgb_0000.png'
image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
image_np = np.array(image)
img_resized = image.resize((416, 416))  # Resize image to YOLOv5 input size
img_resized_np = np.array(img_resized).astype(np.float32)
img_resized_np = np.transpose(img_resized_np, (2, 0, 1))  # Convert to (3, 416, 416)
img_resized_np = np.expand_dims(img_resized_np, axis=0)  # Add batch dimension
img_resized_np /= 255.0  # Normalize to [0, 1]

# Ensure input tensor shape matches the model's expected input
input_name = ort_session.get_inputs()[0].name

# Measure inference time
start_time = time.time()
outputs = ort_session.run(None, {input_name: img_resized_np})
inference_time = time.time() - start_time
predictions = torch.tensor(outputs[0])  # Convert the output to a torch tensor

# Filter out low-confidence predictions
predictions = predictions[0]  # Remove batch dimension
objectness_threshold = 0.3
predictions = predictions[predictions[:, 4] > objectness_threshold]  # Apply objectness threshold

# Extract boxes, scores, and labels
boxes = predictions[:, :4]  # x_center, y_center, width, height
objectness = predictions[:, 4]  # objectness score
class_scores = predictions[:, 5:]  # class scores
labels = class_scores.argmax(dim=1)  # Get class labels

# Convert from center-width-height to x1, y1, x2, y2
boxes = boxes.clone()
boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2
boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2

# Scale boxes back to the original image size
boxes = scale_coords((416, 416), boxes, image_np.shape).numpy()

# Draw bounding boxes on the image
for box, label in zip(boxes, labels):
    x1, y1, x2, y2 = map(int, box)
    color = class_colors[label]  # Get color based on class
    cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)  # Draw box
    cv2.putText(image_np, class_names[label], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Label box

# Convert the image to RGB for displaying with matplotlib
image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

# Display the image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(image_np_rgb)
plt.axis('off')
plt.show()

# Print inference time
print(f"Inference Time: {inference_time:.4f} seconds")

# Read ground truth labels from YOLO format file
label_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\yolov5_format\train\labels\rgb_0000.txt'
ground_truth_boxes = read_yolo_labels(label_path)

# Convert ground truth from center-width-height to x1, y1, x2, y2
ground_truth_boxes[:, 1] = ground_truth_boxes[:, 1] * image_np.shape[1] - ground_truth_boxes[:, 3] * image_np.shape[1] / 2  # x1
ground_truth_boxes[:, 2] = ground_truth_boxes[:, 2] * image_np.shape[0] - ground_truth_boxes[:, 4] * image_np.shape[0] / 2  # y1
ground_truth_boxes[:, 3] = ground_truth_boxes[:, 1] + ground_truth_boxes[:, 3] * image_np.shape[1]  # x2
ground_truth_boxes[:, 4] = ground_truth_boxes[:, 2] + ground_truth_boxes[:, 4] * image_np.shape[0]  # y2

# Calculate accuracy and IoU
correct_predictions = 0
iou_scores = []

for gt in ground_truth_boxes:
    gt_label = int(gt[0])
    gt_box = gt[1:]
    max_iou = 0
    max_iou_label = None
    
    for box, label in zip(boxes, labels):
        iou = calculate_iou(gt_box, box)
        if iou > max_iou:
            max_iou = iou
            max_iou_label = label
    
    iou_scores.append(max_iou)
    if max_iou_label == gt_label:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / len(ground_truth_boxes)

# Calculate mean IoU
mean_iou = np.mean(iou_scores)



