import onnx
import onnxruntime as ort

# Load the ONNX model
model_path = "models/yolov5.onnx"  # or "models/yolov5.onnx"
model = onnx.load(model_path)
onnx.checker.check_model(model)

# Print input and output names and shapes
print("Inputs:")
for input in model.graph.input:
    print(f"Name: {input.name}, Shape: {input.type.tensor_type.shape}")

print("\nOutputs:")
for output in model.graph.output:
    print(f"Name: {output.name}, Shape: {output.type.tensor_type.shape}")

# Alternatively, use onnxruntime to get this info
ort_session = ort.InferenceSession(model_path)
print("\nInput names:", ort_session.get_inputs())
print("Output names:", ort_session.get_outputs())
