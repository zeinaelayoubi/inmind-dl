import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

#leNet5 CNN implementation
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
   
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,stride=1)
    
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,stride=1)
      
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5,stride=1)
        
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)
        
    def forward(self, x):
#       Convolutional Layer: Applies filters (kernels) to the input data to extract features.
#       Activation Function: Introduces non-linearity to the model, allowing it to learn complex patterns.
#       Average Pooling: Reduces the spatial dimensions (height and width) of the feature maps, helping to make the model more computationally efficient and robust to spatial variations.
        x = self.conv1(x)
        x = torch.tanh(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
       
        x = self.conv2(x)
        x = torch.tanh(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        
        x = self.conv3(x)
        x = torch.tanh(x)
        
        #flatten out to the shape of (btch size, 120)
        x = x.view(-1, 120)
        
        x = self.fc1(x)
        x = torch.tanh(x)
        
       
        x = self.fc2(x)
        x = F.softmax(x, dim=1)#usually always dim=1
        
        return x


model = LeNet5()
print(model)

input_tensor = torch.randn(1, 1, 32, 32)  # (batch_size, channels, height, width)

output = model(input_tensor)

# Print the input and output
print('Input Tensor:')
print(input_tensor)
print('\nOutput Tensor:')
print(output)
print(output.shape)
summary(model, input_size=(1, 32, 32)) 
