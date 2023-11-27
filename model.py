# deep learning model for pneumonia detection
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights

class PneumoniaDetectionResNet18Model(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.transforms = weights.transforms(antialias=True)
        self.network = resnet18(weights=weights, progress=False)

        num_features = self.network.fc.in_features # get number of in features of last layer
        self.network.fc = nn.Linear(num_features, 2) # replace model classifier
        self.history = {}

    def forward(self, x):
        x = self.transforms(x)
        return self.network(x)
    
    def set_history(self, history):
        self.history = history
    
    def string(self):
        return self.network.string()
    
class PneumoniaDetectionMobileNetV3LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT
        self.transforms = weights.transforms(antialias=True)
        self.network = mobilenet_v3_large(weights=weights, progress=False)

        num_features = self.network.classifier[3].in_features # get number of in features of last layer
        self.network.classifier[3] = nn.Linear(num_features, 2) # replace model classifier
        self.history = {}

    def forward(self, x):
        x = self.transforms(x)
        return self.network(x)
    
    def set_history(self, history):
        self.history = history
    
    def string(self):
        return self.network.string()
