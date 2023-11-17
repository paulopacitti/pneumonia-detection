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

transform_res_net_18 = transforms.Compose([
    transforms.Resize((256, 256), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.Normalize(mean, std),
])

transform_crop_and_resize = transforms.Compose([
    transforms.Resize((256, 256), antialias=True),
    transforms.CenterCrop(256),
])

transform_to_image = transforms.Compose([
    transforms.ToPILImage(),
])

def split_dataset(dataset, split_ratio=0.8):
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def plot_raw_transfomed_examples(dataloader, classes, transform):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    transformed_images = transform(images)

    figure = plt.figure(figsize=(12, 7))
    cols, rows = 4,2
    for i in range(1, cols * rows + 1):
        if i <= 4:
            figure.add_subplot(rows, cols, i)
            plt.title(classes[labels[i]])
            plt.xlabel("raw")
            plt.imshow(transform_to_image(images[i]))
        else:
            figure.add_subplot(rows, cols, i)
            plt.title(classes[labels[i]])
            plt.xlabel("transformed")
            plt.imshow(transform_to_image(transformed_images[i]))
    plt.show()