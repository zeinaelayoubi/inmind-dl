import torch
import torch.onnx
from model import SimpleSegNet

def export_model_to_onnx(model_path, onnx_file_path, input_size):
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model_path (str): Path to the PyTorch model weights file.
        onnx_file_path (str): Path where the ONNX model file will be saved.
        input_size (tuple): Size of the input tensor (height, width).
    """
    # Initialize the model
    num_classes = 11  # Adjust this based on your model's configuration
    model = SimpleSegNet(num_classes=num_classes)
    
    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])  # Adjust key if necessary
    
    # Set the model to evaluation mode
    model.eval()

    # Create a sample input tensor with the same size as your input images
    sample_input = torch.randn(1, 3, *input_size)  # Batch size of 1, RGB channels, height x width

    # Export the model to ONNX
    torch.onnx.export(
        model,               # The model to export
        sample_input,        # A sample input tensor
        onnx_file_path,      # File path to save the ONNX model
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=12,    # ONNX version to use
        do_constant_folding=True,  # Whether to apply optimizations (optional)
        input_names=['input'],    # Names of input tensors
        output_names=['output'],  # Names of output tensors
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic axes (optional)
    )

if __name__ == "__main__":
    # Paths
    model_weights_path = 'best_model_weights.pt'
    onnx_model_path = 'seg_model.onnx'
    
    # Input size (height, width) of your images
    input_height = 720
    input_width = 1280
    input_size = (input_height, input_width)

    # Export the model
    export_model_to_onnx(model_weights_path, onnx_model_path, input_size)
