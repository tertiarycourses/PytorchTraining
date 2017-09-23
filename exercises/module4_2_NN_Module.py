# Deep Learning with Pytorch
# Module 4: Neural Network with Pytorch
# NN Module

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
import matplotlib.pyplot as plt

# Step 1: Setup
torch.manual_seed(1)

# Hyper parameters
learning_rate = 0.05

# Variables
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  
y = x + 0.2*torch.rand(x.size())                 

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# Step 2: Model
class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,10)    
        self.fc2 = nn.Linear(10,1)

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)             
        return x

model = Model() 
print(model) 

# Step 3: Loss function
loss_func = nn.MSELoss()  

# Step 4: Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(100):

	# Model prediction
    yhat = model(x)     

    # Compute loss
    loss = loss_func(yhat, y)
    #loss = F.mse_loss(yhat,y)

    # Compute gradients and update parameters     
    optimizer.zero_grad()   
    loss.backward()         
    optimizer.step()       

    if i % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), yhat.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0])
        plt.pause(0.1)

plt.show()