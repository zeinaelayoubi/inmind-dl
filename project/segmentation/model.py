import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class HRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, stride=1):
        super(HRBlock, self).__init__()
        self.layers = nn.Sequential(
            Bottleneck(in_channels, out_channels, stride),
            *[Bottleneck(out_channels * 4, out_channels) for _ in range(1, blocks)]
        )

    def forward(self, x):
        return self.layers(x)

class HRNetV2Inspired(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(HRNetV2Inspired, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.hr_stage1 = HRBlock(64, 64, blocks=2)      # Simplified blocks
        self.hr_stage2 = HRBlock(64 * 4, 128, blocks=2) # Simplified blocks
        self.hr_stage3 = HRBlock(128 * 4, 256, blocks=2) # Simplified blocks
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256 + 512 + 1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.segmentation_head = nn.Conv2d(512, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        
        x = self.initial_conv(x)
        #print(f"After initial_conv: {x.shape}")

        x1 = self.hr_stage1(x)
        #print(f"After hr_stage1: {x1.shape}")

        x2 = self.hr_stage2(x1)
        #print(f"After hr_stage2: {x2.shape}")

        x3 = self.hr_stage3(x2)
        #print(f"After hr_stage3: {x3.shape}")

        x1_resized = TF.resize(x1, size=x3.shape[2:])
        x2_resized = TF.resize(x2, size=x3.shape[2:])
        #print(f"After resizing x1: {x1_resized.shape}")
        #print(f"After resizing x2: {x2_resized.shape}")

        fused_features = torch.cat([x1_resized, x2_resized, x3], dim=1)
        #print(f"After concatenation: {fused_features.shape}")

        fused_features = self.fusion_conv(fused_features)
        #print(f"After fusion_conv: {fused_features.shape}")

        output = self.upsample(fused_features)
        #print(f"After upsampling: {output.shape}")
        
        output = self.segmentation_head(output)
        #print(f"Output shape: {output.shape}")
        
        return output

def test():
    x = torch.randn((1, 3, 256, 256))  # Batch size of 1, input shape 512x512
    model = HRNetV2Inspired(in_channels=3, num_classes=11)  # Adjust num_classes to match your dataset
    preds = model(x)
    assert preds.shape == (1, 11, 256, 256), f"Unexpected output shape: {preds.shape}"
    print("Shape verification passed!")

if __name__ == "__main__":
    test()
