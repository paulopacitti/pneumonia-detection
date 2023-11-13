# load and transform dataset

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, random_split

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform_to_tensor = transforms.Compose([
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Resize((224, 224), antialias=True),  
    transforms.CenterCrop(224),
])

transform_normalize = transforms.Compose([
    transforms.Normalize(mean=mean, std=std),
])

transform_to_image = transforms.Compose([
    transforms.ToPILImage(),
])

def split_dataset(dataset, split_ratio=0.8):
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset