# Deep Learning with Pytorch
# Module 4: Neural Network with Pytorch
# Dataset

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Step 1: Setup
torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 2               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate

# Mnist digits dataset
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose(
    [transforms.ToTensor()])

train_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=True, 
    download=True, 
    transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2)

test_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=False,
    download=True, 
    transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2)

# Step 2: Model
L1 = 200
L2 = 100
L3 = 60
L4 = 30


model = nn.Sequential(
    nn.Linear(784, L1),
    nn.ReLU(),
    nn.Linear(L1, L2),
    nn.ReLU(),
    nn.Linear(L2, L3),
    nn.ReLU(),
    nn.Linear(L3, L4),
    nn.ReLU(),
    nn.Linear(L4, 10),
    nn.Softmax(),
)

# Step 3: Loss Function
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# Step 4: Optmizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all  parameters

correct = 0
total = 0
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x.view(-1, 784))      
        b_y = Variable(y) 

        b_yhat = model(b_x)
        loss = loss_func(b_yhat, b_y)    # cross entropy loss

        optimizer.zero_grad()            # clear gradients for this training step
        loss.backward()                  # backpropagation, compute gradients
        optimizer.step()                 # apply gradients

        _, y_pred = torch.max(b_yhat.data, 1)
        total += y.size(0)
        correct += (y_pred == y).sum()
        if i % 20 == 0:
            print('Epoch/Step: ', epoch, '/',i, 
                '| train loss: %.4f' % loss.data[0], 
                '| accuracy: %.2f %%' % (100 * correct / total))

correct = 0
total = 0
for (x,y) in test_loader:
    b_x = Variable(x.view(-1, 784))           
    b_y = Variable(y)

    yhat = model(b_x)
    _, y_pred = torch.max(yhat.data, 1)
    total += y.size(0)
    correct += (y_pred == y).sum()
print('Test accuracy: %.2f %%' % (100 * correct / total))


