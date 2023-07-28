import torch
import torch.nn as nn

from layer import *

class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, architecture="VGG16"):
        super(VGG, self).__init__()

        self.conv_layers = Conv(in_channels=in_channels, architecture=architecture)
        self.fc_layers = FC(num_classes=num_classes)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        return x

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = VGG().to(device)
x = torch.randn(1, 3, 224, 224).to(device)
print(model(x).shape)
        
