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

app = FastAPI()

# Model paths
ONNX_SEGMENTATION_MODEL_PATH = "models/seg_model.onnx"
ONNX_OBJECT_DETECTION_MODEL_PATH = "models/yolov5.onnx"

# Color map for segmentation
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

# Function to colorize segmentation mask
def colorize_segmentation(mask: np.ndarray, color_map: Dict[int, List[int]]) -> np.ndarray:
    colored_image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    for label, color in color_map.items():
        colored_image[mask == label] = color
    return colored_image

# Define object detection class names and colors
class_names = [
    "forklift", "rack", "crate", "floor", "railing", 
    "pallet", "stillage", "iwhub", "dolly"
]
class_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0),
    (0, 128, 128)
]

# Endpoint for listing available models
@app.get("/models")
def list_models():
    return {"models": ["segmentation", "object_detection"]}

# Endpoint for segmentation inference
@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        ort_session = ort.InferenceSession(ONNX_SEGMENTATION_MODEL_PATH)
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize([720, 1280]),  
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0).numpy()
        start_time = time.time()
        outputs = ort_session.run(None, {'input': image_tensor})[0]
        inference_time = time.time() - start_time
        pred_mask = np.argmax(outputs, axis=1)[0]
        colored_mask = colorize_segmentation(pred_mask, COLOR_MAP)
        result_image = Image.fromarray(colored_mask, mode='RGBA')
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return {
            "inference_time": inference_time,
            "image": img_byte_arr.getvalue()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for object detection with bounding boxes
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        ort_session = ort.InferenceSession(ONNX_OBJECT_DETECTION_MODEL_PATH)
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        img_resized = image.resize((416, 416))
        img_resized_np = np.array(img_resized).astype(np.float32)
        img_resized_np = np.transpose(img_resized_np, (2, 0, 1))
        img_resized_np = np.expand_dims(img_resized_np, axis=0)
        img_resized_np /= 255.0
        start_time = time.time()
        outputs = ort_session.run(None, {'input': img_resized_np})[0]
        inference_time = time.time() - start_time
        predictions = torch.tensor(outputs)
        objectness_threshold = 0.3
        predictions = predictions[predictions[:, 4] > objectness_threshold]
        boxes = predictions[:, :4].clone()
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        boxes = scale_coords((416, 416), boxes, image_np.shape).numpy()
        for box, prediction in zip(boxes, predictions):
            x1, y1, x2, y2 = map(int, box)
            label_idx = int(prediction[5:].argmax())
            label = class_names[label_idx] if 0 <= label_idx < len(class_names) else "Unknown"
            color = class_colors[label_idx] if 0 <= label_idx < len(class_colors) else (255, 0, 0)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        result_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return {
            "inference_time": inference_time,
            "image": img_byte_arr.getvalue()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to scale coordinates
def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2
    coords[:, [0, 2]] = (coords[:, [0, 2]] - pad) / gain
    coords[:, [1, 3]] = (coords[:, [1, 3]] - pad) / gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords
