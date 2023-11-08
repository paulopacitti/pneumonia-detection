# load and transform dataset

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.CenterCrop(224),
    transforms.ToDtype(torch.uint8, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # RGB mean and std parameters
])

train_set = torchvision.datasets.ImageFolder(root="data/train", transform=transform_train)
validation_set = torchvision.datasets.ImageFolder(root="data/val", transform=transform_train)
test_set = torchvision.datasets.ImageFolder(root="data/test", transform=transform_train)

batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
