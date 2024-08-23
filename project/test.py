import torch
import sys

print("PyTorch version:", torch.__version__)
# print("Python version:", sys.version)
# print("CUDA available:", torch.cuda.is_available())

import torch
print(torch.cuda.device_count())

import torch
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())
import torch
torch.cuda.empty_cache()
 
import sys
print(sys.version)

 