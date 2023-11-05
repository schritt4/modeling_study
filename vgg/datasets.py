import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def Dataset(root):
    transform = transforms.Compose( [
                        transforms.Resize(250), 
                        transforms.CenterCrop(224),
                        
                        transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        
    return torchvision.datasets.ImageFolder(root=root, transform=transform)

