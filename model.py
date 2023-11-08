# deep learning model for pneumonia detection

import torch.nn as nn

class PneumoniaDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x