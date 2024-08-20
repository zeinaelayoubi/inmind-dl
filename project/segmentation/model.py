import torch
import torch.nn as nn

class SimpleSegNet(nn.Module):
    def __init__(self, num_classes=11):
        super(SimpleSegNet, self).__init__()
        
        # Encoder
        self.enc1 = self._block(3, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        # Decoder
        self.dec4 = self._up_block(512, 256)
        self.dec3 = self._up_block(256, 128)
        self.dec2 = self._up_block(128, 64)
        self.dec1 = self._up_block(64, 64)  # Output channels match input to final layer
        
        # Final segmentation layer
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
        #print("Input shape:", x.shape)

        # Encoder
        enc1 = self.enc1(x)
        #print("After enc1 (64 channels):", enc1.shape)

        enc2 = self.enc2(enc1)
        #print("After enc2 (128 channels):", enc2.shape)

        enc3 = self.enc3(enc2)
        #print("After enc3 (256 channels):", enc3.shape)

        #enc4 = self.enc4(enc3)
        #print("After enc4 (512 channels):", enc4.shape)

        # Decoder
        #dec4 = self.dec4(enc4)
        #print("After dec4 (256 channels):", dec4.shape)

        dec3 = self.dec3(enc3)
        #print("After dec3 (128 channels):", dec3.shape)

        dec2 = self.dec2(dec3)
        #print("After dec2 (64 channels):", dec2.shape)

        dec1 = self.dec1(dec2)
        #print("After dec1 (64 channels):", dec1.shape)

        # Final layer
        output = self.final(dec1)
        #print("Final output shape:", output.shape)

        return output


if __name__ == "__main__":
    model = SimpleSegNet(num_classes=11)
    input_tensor = torch.randn(1, 3, 640, 640)  # Batch size of 1, 3 channels, 720x1280 image
    output = model(input_tensor)
    print("Output shape:", output.shape)
