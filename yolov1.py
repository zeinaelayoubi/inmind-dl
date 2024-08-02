import torch
import torch.nn as nn
      
        
class miniblock(nn.Module):
    def __init__(self,in_channels,num_filters,kernel_size,stride,padding):
        super(miniblock, self).__init__()
        self.conv=nn.Conv2d(in_channels,num_filters,bias=False,kernel_size=kernel_size,stride=stride, padding=padding)
        self.batchnorm=nn.BatchNorm2d(num_filters)
        self.leakyrelu=nn.LeakyReLU(0.1)
        
        
    def forward(self,x):
        x=self.conv(x)
        x=self.batchnorm(x)
        x=self.leakyrelu(x)
        return x
        

class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.conv_layers = nn.Sequential(
            miniblock(3, 64, kernel_size=7, stride=2, padding=3), #input , output(or filters),kernel size,stride, padding
            nn.MaxPool2d(kernel_size=2, stride=2),
            miniblock(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            miniblock(192, 128, kernel_size=1, stride=1, padding=0),
            miniblock(128, 256, kernel_size=3, stride=1, padding=1),
            miniblock(256, 256, kernel_size=1, stride=1, padding=0),
            miniblock(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *[nn.Sequential(
                miniblock(512, 256, kernel_size=1, stride=1, padding=0),
                miniblock(256, 512, kernel_size=3, stride=1, padding=1)
            ) for _ in range(4)],
            miniblock(512, 512, kernel_size=1, stride=1, padding=0),
            miniblock(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *[nn.Sequential(
                miniblock(1024, 512, kernel_size=1, stride=1, padding=0),
                miniblock(512, 1024, kernel_size=3, stride=1, padding=1)
            ) for _ in range(2)],
            miniblock(1024, 1024, kernel_size=3, stride=1, padding=1),
            miniblock(1024, 1024, kernel_size=3, stride=2, padding=1),
            miniblock(1024, 1024, kernel_size=3, stride=1, padding=1),
            miniblock(1024, 1024, kernel_size=3, stride=1, padding=1)#output of this is 7*7*1024
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x


model = YOLOv1(S=7, B=2, C=20)  # Adjust S, B, and C if needed

input_tensor = torch.randn(1, 3, 448, 448)  # (batch_size, channels, height, width)

model.eval()  
with torch.no_grad():
    output = model(input_tensor)

print('Output Tensor Shape:', output.shape)
print(output)
#the output is 30 values of each pixel containing x,y,w,h,confidence and we have 2 bounding box and we have 20 classes 
#30 times 49 pixels is 1470

