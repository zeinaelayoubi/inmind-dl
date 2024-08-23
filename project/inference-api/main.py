from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List, Dict
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import time
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

app = FastAPI()

# Model paths
ONNX_OBJECT_DETECTION_MODEL_PATH = "models/yolov5.onnx"
ONNX_SEGMENTATION_MODEL_PATH = "models/seg_model.onnx"

# Load the ONNX model
ort_session = ort.InferenceSession(ONNX_OBJECT_DETECTION_MODEL_PATH)
ort_session_segmentation = ort.InferenceSession(ONNX_SEGMENTATION_MODEL_PATH)

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

COLOR_MAP = {
    0: (0, 0, 0, 0),
    1: (25, 255, 82, 255),
    2: (25, 82, 255, 255),
    3: (255, 25, 197, 255),
    4: (140, 25, 255, 255),
    5: (140, 255, 25, 255),
    6: (255, 111, 25, 255),
    7: (0, 0, 0, 255),
    8: (226, 255, 25, 255),
    9: (255, 197, 25, 255),
    10: (54, 255, 25, 255)
}

def colorize_segmentation(pred_mask_np, color_map):
    color_mask = np.zeros((pred_mask_np.shape[0], pred_mask_np.shape[1], 4), dtype=np.uint8)
    for class_index, color in color_map.items():
        color_mask[pred_mask_np == class_index] = color
    return color_mask

# Endpoint for listing available models
@app.get("/models")
def list_models():
    return {"models": ["segmentation", "object_detection"]}

# Endpoint for object detection with bounding boxes
@app.post("/bbox-image")
async def detect_objects(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')  # Ensure image is in RGB mode
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
    label_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\yolov5_format\val\labels\rgb_0059.txt'
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

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    

    return {
        "inference_time": inference_time,
        "accuracy": accuracy,
    }

@app.post("/bbox-json")
async def detect_objects_json(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')  # Ensure image is in RGB mode
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

    # Prepare the response JSON
    response = []
    for box, label, class_score in zip(boxes, labels, class_scores):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[label]
        accuracy = float(class_score[label])
        response.append({
            "class": class_name,
            "accuracy": accuracy,
            "bounding_box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        })

    return response




# Endpoint for segmentation inference
@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')  # Ensure image is in RGB mode

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize([720, 1280]),  # Ensure the image size matches the model's input size
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).numpy()  # Convert to numpy array

    # Perform inference and time it
    start_time = time.time()  # Start timing
    outputs = ort_session_segmentation.run(None, {'input': image})  # Run inference
    end_time = time.time()  # End timing

    # Calculate the elapsed time
    inference_time = end_time - start_time
    print(f'Inference time: {inference_time:.4f} seconds')

    # Get the prediction by taking the class with the highest probability
    pred_mask = np.argmax(outputs[0], axis=1)[0]  # Outputs shape: [batch_size, num_classes, height, width]

    # Debug: Print statistics of the predicted mask
    print(f'Predicted mask shape: {pred_mask.shape}')
    print(f'Predicted mask unique values: {np.unique(pred_mask)}')

    # Apply color map to the predicted mask
    colored_mask = colorize_segmentation(pred_mask, COLOR_MAP)

    # Convert the colored mask to a PIL image
    colored_mask_pil = Image.fromarray(colored_mask, mode='RGBA')

    # Save or display the predicted mask
    colored_mask_pil.save('inferred_colored_mask.png')
    colored_mask_pil.show()

    return {
        "inference_time": inference_time,
        "predicted_mask_shape": pred_mask.shape,
        "unique_values": np.unique(pred_mask).tolist()
    }