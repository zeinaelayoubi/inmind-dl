import torch
import sys

print("PyTorch version:", torch.__version__)
print("Python version:", sys.version)
print("CUDA available:", torch.cuda.is_available())
