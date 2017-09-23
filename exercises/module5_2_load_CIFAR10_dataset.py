# Deep Learning with Pytorch
# Module 5: Convolutional Neural Networks (CNN)
# Load CIFAR10 datasdet

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt

train_data = torchvision.datasets.CIFAR10(
    root='./cifar10/', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=128, 
    shuffle=True)

test_data = torchvision.datasets.CIFAR10(
    root='./cifar10/', 
    train=False,
    download=True, 
    transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(
    test_data, 
    batch_size=128, 
    shuffle=False)
