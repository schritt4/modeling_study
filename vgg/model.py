import torch
import torch.nn as nn

from layer import *

class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, architecture="VGG16"):
        super(VGG, self).__init__()
        self.num_classes = num_classes

        self.conv_layers = Conv(in_channels=in_channels, architecture=architecture)
        if num_classes == 2:
            self.fc_layers = FC(num_classes=1)
        else:
            self.fc_layers = FC(num_classes=num_classes)
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        return x

        
