# deep learning model for pneumonia detection
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class PneumoniaDetectionResNetModel(nn.Module):
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
