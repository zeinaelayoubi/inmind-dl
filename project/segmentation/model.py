import torch
import torch.nn as nn

class SimpleSegNet(nn.Module):
    def __init__(self, num_classes=11):
        super(SimpleSegNet, self).__init__()
        
        self.enc1 = self._block(3, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        #self.enc4 = self._block(256, 512)

        #self.dec4 = self._up_block(512, 256)
        self.dec3 = self._up_block(256, 128)
        self.dec2 = self._up_block(128, 64)
        self.dec1 = self._up_block(64, 64)
        
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        #enc4 = self.enc4(enc3)
        #dec4 = self.dec4(enc4)
        dec3 = self.dec3(enc3)
        dec2 = self.dec2(dec3)
        dec1 = self.dec1(dec2)
        output = self.final(dec1)
        return output

if __name__ == "__main__":
    model = SimpleSegNet(num_classes=11)
    input_tensor = torch.randn(1, 3, 720, 1280)
    output = model(input_tensor)
    print("Output shape:", output.shape)
