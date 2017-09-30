# Deep Learning with Pytorch
# Module 4: Neural Network with Pytorch
# Save Model and parameters

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn

# Step 1: Setup
torch.manual_seed(1)    

# Hyper Parameters
EPOCH = 2               
BATCH_SIZE = 50
LR = 0.001              

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


# Step 2: Model
L1 = 200
L2 = 100
L3 = 60
L4 = 30


model = torch.nn.Sequential(
    torch.nn.Linear(784, L1),
    torch.nn.ReLU(),
    torch.nn.Linear(L1, L2),
    torch.nn.ReLU(),
    torch.nn.Linear(L2, L3),
    torch.nn.ReLU(),
    torch.nn.Linear(L3, L4),
    torch.nn.ReLU(),
    torch.nn.Linear(L4, 10),
)

# Step 3: Loss Function
loss_func = nn.CrossEntropyLoss()                       

# Step 4: Optmizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all  parameters

correct = 0
total = 0
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        x = Variable(x.view(-1, 784))      
        y = Variable(y) 

        yhat = model(x)
        loss = loss_func(yhat, y)    # cross entropy loss

        optimizer.zero_grad()            # clear gradients for this training step
        loss.backward()                  # backpropagation, compute gradients
        optimizer.step()                 # apply gradients

        _, y_pred = torch.max(yhat.data, 1)
        total += y.size(0)
        correct += (y_pred == y.data).sum()
        if i % 20 == 0:
            print('Epoch/Step: ', epoch, '/',i, 
                '| train loss: %.4f' % loss.data[0], 
                '| accuracy: %.2f %%' % (100 * correct / total))

torch.save(model, 'mnist.pkl')  # save entire net
#torch.save(net1.state_dict(), 'mnist_params.pkl')   # save only the parameters


