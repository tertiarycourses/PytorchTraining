# Deep Learning with Pytorch
# Module 7: (Recurrent Neural Network) RNN with Pytorch

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
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 128
RNN_SIZE = 100          # rnn hidden units
INPUT_SIZE = 32         # rnn input size / image width
LR = 0.01               # learning rate

train_data = torchvision.datasets.CIFAR10(
    root='./cifar10/', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2)

test_data = torchvision.datasets.CIFAR10(
    root='./cifar10/', 
    train=False,
    download=True, 
    transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(
    test_data, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Step 2: Model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=RNN_SIZE,   # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc = nn.Linear(RNN_SIZE, 10)

    def forward(self, x):
        x, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        x = self.fc(x[:, -1, :])
        return x


rnn = RNN()
print(rnn)

# Step 3: Loss Function
loss_func = nn.CrossEntropyLoss()                      

# Step 4: Optmizer
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   


# Step 5: Training Loop
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):   
        b_x = Variable(x.view(-1, 32,32))      
        b_y = Variable(y)

        b_yhat = rnn(b_x)
        loss = loss_func(b_yhat, b_y)    # cross entropy loss

        optimizer.zero_grad()            # clear gradients for this training step
        loss.backward()                  # backpropagation, compute gradients
        optimizer.step()                 # apply gradients

        _, y_pred = torch.max(b_yhat.data, 1)
        total = y.size(0)
        correct = (y_pred == y).sum()
        if i % 10 == 0:
            print('Epoch/Step: ', epoch, '/',i, 
                '| train loss: %.4f' % loss.data[0], 
                '| accuracy: %.2f %%' % (100 * correct / total))

#Step 6: Evaluation
for (x,y) in test_loader:
    b_x = Variable(x.view(-1, 32, 32,3))           
    b_y = Variable(y)

    yhat = rnn(b_x)
    _, y_pred = torch.max(yhat.data, 1)
    total = y.size(0)
    correct = (y_pred == y).sum()
print('Test accuracy: %.2f %%' % (100 * correct / total))

