# Deep Learning with Pytorch
# Module 5: Convolutional Neural Networks (CNN)
# MNIST dataset

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
EPOCH = 1               
BATCH_SIZE = 128
LR = 0.01             

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2)

test_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=False,
    download=True, 
    transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(
    test_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2)

# Step 2: Model
#class CNN(nn.Module):
    # def __init__(self):
    #     super(CNN, self).__init__()
    #     self.conv1 = nn.Sequential(             # input shape (1, 28, 28)
    #         nn.Conv2d(1,16,5,1,2),
    #         nn.ReLU(),                      
    #         nn.MaxPool2d(2)                     # output shape (16, 14, 14)
    #     )
    #     self.conv2 = nn.Sequential(             # input shape (1, 28, 28)
    #         nn.Conv2d(16, 32, 5, 1, 2),         # output shape (32, 14, 14)
    #         nn.ReLU(),                          # activation
    #         nn.MaxPool2d(2)                     # output shape (32, 7, 7)
    #     )
    #     self.fc = nn.Linear(32 * 7 * 7, 10)     # fully connected layer, output 10 classes

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = x.view(x.size(0), -1)               # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
    #     x = self.fc(x)
    #     return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,16,5,1,2)
        self.conv2 = nn.Conv2d(16,32,5,1,2)
        #self.dropout = nn.Dropout2d()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        #x = self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size(0), -1)                       
        x = self.fc(x)
        return x


cnn = CNN()
print(cnn) 

# Step 3: Loss Function
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# Step 4: Optmizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters

# Step 5: Training Loop
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        x, y = Variable(x), Variable(y)  

        yhat = cnn(x)
        loss = loss_func(yhat, y)    # cross entropy loss

        optimizer.zero_grad()            # clear gradients for this training step
        loss.backward()                  # backpropagation, compute gradients
        optimizer.step()                 # apply gradients

        _, y_pred = torch.max(yhat.data, 1)
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
