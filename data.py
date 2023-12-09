# load and transform dataset
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform_to_tensor = transforms.Compose([
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
])

transform_crop_and_resize = transforms.Compose([
    transforms.Resize((256, 256), antialias=True),
    transforms.CenterCrop(256),
])

transform_to_image = transforms.Compose([
    transforms.ToPILImage(),
])

transform_res_net_18 = transforms.Compose([
    transforms.Resize((256, 256), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.Normalize(mean, std),
])

transform_mnasnet_0_75 = transforms.Compose([
    transforms.Resize((232, 232), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.Normalize(mean, std),
])

def split_dataset(dataset, split_ratio=0.8):
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def plot_examples(caption, images, labels, classes, transform):
    if transform:
        images = transform(images[:4])
    
    figure = plt.figure(figsize=(12, 7))
    rows, columns = 1, 4
    for i in range(1, columns + 1):
        figure.add_subplot(rows, columns, i)
        plt.title(classes[labels[i-1]])
        plt.xlabel(caption)
        plt.imshow(transform_to_image(images[i-1]))