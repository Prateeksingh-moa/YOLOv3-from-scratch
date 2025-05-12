import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels//2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=channels//2, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.layers = nn.ModuleList()
        self.in_channels = 32
        self._create_conv_layers()
    
    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    
    def _make_layer(self, out_channels, num_blocks):
        layers = []
        
        # Downsample
        layers.append(self._conv_block(self.in_channels, out_channels, 3, 2, 1))
        self.in_channels = out_channels
        
        # Residual Blocks
        for _ in range(num_blocks):
            layers.append(ResidualBlock(self.in_channels))
        
        return nn.Sequential(*layers)
    
    def _create_conv_layers(self):
        self.layers.append(self._conv_block(3, 32, 3, 1, 1))  # Initial Conv layer
        
        # Creates the full Darknet-53 structure
        self.layers.append(self._make_layer(64, 1))    # Layer 1
        self.layers.append(self._make_layer(128, 2))   # Layer 2
        self.layers.append(self._make_layer(256, 8))   # Layer 3
        self.layers.append(self._make_layer(512, 8))   # Layer 4
        self.layers.append(self._make_layer(1024, 4))  # Layer 5
    
    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            outputs.append(x)
        return outputs  # returns all intermediate outputs for multi-scale detection