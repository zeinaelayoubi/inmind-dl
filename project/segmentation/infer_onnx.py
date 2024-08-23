import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import time  # Import the time module
from utils import colorize_segmentation, COLOR_MAP  # Import colorize_segmentation and COLOR_MAP

def inference(image_path, onnx_model_path):
    # Load the ONNX model
    ort_session = ort.InferenceSession(onnx_model_path)

    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize([720, 1280]),  # Ensure the image size matches the model's input size
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).numpy()  # Convert to numpy array

    # Perform inference and time it
    start_time = time.time()  # Start timing
    outputs = ort_session.run(None, {'input': image})  # Run inference
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

# Example usage
image_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\train\temp_images\rgb_0018.png'
onnx_model_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\segmentation\seg_model.onnx'
inference(image_path, onnx_model_path)
