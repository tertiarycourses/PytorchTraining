# Deep Learning with Pytorch
# Module 4: Neural Network with Pytorch
# Load MNIST dataset

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

train_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=128, 
    shuffle=True)

test_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=False,
    download=True, 
    transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(
    test_data, 
    batch_size=128, 
    shuffle=True)

# print('Training dataset size: ',train_data.train_data.size())                 
# print('Training label size: ',train_data.train_labels.size())               
# print('Trainning batches: ',len(train_loader))
# print(train_data.train_data[0].numpy())
# print(train_data.train_labels[0])

# plot one example
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

