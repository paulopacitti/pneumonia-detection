# deep learning model for pneumonia detection
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights, mnasnet0_75, MNASNet0_75_Weights

class PneumoniaDetectionResNet18Model(nn.Module):
    def __init__(self, history=None):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.transforms = weights.transforms(antialias=True)
        self.backbone = resnet18(weights=weights, progress=False)

        num_features = self.backbone.fc.in_features # get number of in features of last layer
        self.backbone.fc = nn.Linear(num_features, 2) # replace model classifier
        self.history = {}

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.transforms(x)
        return self.backbone(x)
    
    def set_history(self, history):
        self.history = history
    
    def string(self):
        return self.backbone.string()
    
class PneumoniaDetectionMobileNetV3LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT
        self.transforms = weights.transforms(antialias=True)
        self.backbone = mobilenet_v3_large(weights=weights, progress=False)

        num_features = self.backbone.classifier[3].in_features # get number of in features of last layer
        self.backbone.classifier[3] = nn.Linear(num_features, 2) # replace model classifier
        self.history = {}

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.transforms(x)
        return self.backbone(x)
    
    def set_history(self, history):
        self.history = history
    
    def string(self):
        return self.backbone.string()
    
class PneumoniaDetectionMNASNet0_75Model(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MNASNet0_75_Weights.DEFAULT
        self.transforms = weights.transforms(antialias=True)
        self.backbone = mnasnet0_75(weights=weights, progress=False)

        # Freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last two layers
        for param in (list(self.backbone.parameters()))[-2:]:
            param.requires_grad = True

        num_features = self.backbone.classifier[1].in_features # get number of in features of last layer
        self.backbone.classifier[1] = nn.Linear(num_features, 2) # replace model classifier

        self.history = {}

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.transforms(x)
        return self.backbone(x)
    
    def set_history(self, history):
        self.history = history
    
    def string(self):
        return self.backbone.string()
