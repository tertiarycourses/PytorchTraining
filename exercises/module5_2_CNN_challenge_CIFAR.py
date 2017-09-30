# Deep Learning with Pytorch
# Module 5: Convolutional Neural Networks (CNN)
# CNN Challenge on CIFAR10 dataset

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Step 1: Setup
torch.manual_seed(1)    

# Hyper Parameters
EPOCH = 2               
BATCH_SIZE = 128
LR = 0.001              

train_data = torchvision.datasets.CIFAR10(
    root='./cifar10', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True)

test_data = torchvision.datasets.CIFAR10(
    root='./cifar10', 
    train=False,
    download=True, 
    transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(
    test_data, 
    batch_size=BATCH_SIZE, 
    shuffle=False)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Step 2: Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


cnn = CNN()
print(cnn)

# Step 3: Loss Funtion
loss_func = nn.CrossEntropyLoss()

#Step 4: Optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

#Step 5: Training Loop
for epoch in range(EPOCH):  # loop over the dataset multiple times
    for i, (x,y) in enumerate(train_loader):
        x, y = Variable(x), Variable(y)  

        yhat = cnn(x)
        loss = loss_func(yhat, y)    # cross entropy loss

        optimizer.zero_grad()            # clear gradients for this training step
        loss.backward()                  # backpropagation, compute gradients
        optimizer.step()                 # apply gradients

        _,y_pred = torch.max(yhat.data, 1)
        total = y.size(0)
        correct = (y_pred == y.data).sum()
        if i % 10 == 0:
            print('Epoch/Step: {}/{}'.format(epoch,i),  
                '| train loss: %.4f' % loss.data[0], 
                '| accuracy: %.2f %%' % (100 * correct / total))

#Step 6: Evaluation
for (x,y) in test_loader:
    yhat = cnn(Variable(x))
    _, y_pred = torch.max(yhat.data, 1)
    total = y.size(0)
    correct = (y_pred == y).sum()
print('Test accuracy: %.2f %%' % (100 * correct / total))

