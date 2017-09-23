# Deep Learning with Pytorch
# Module 4: Regression with Pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
import matplotlib.pyplot as plt

# Generate data
x = torch.unsqueeze(torch.linspace(-4, 2, 100), dim=1)  
y = x*x + 2*x + 1 + 0.5*torch.rand(x.size())

# plt.scatter(x.numpy(),y.numpy())
# plt.show()

# Step 1: Setup
x, y = Variable(x), Variable(y)

W1 = Variable(torch.randn(1), requires_grad=True)
W2 = Variable(torch.randn(1), requires_grad=True)
b = Variable(torch.randn(1), requires_grad=True)

# Step 2: Optimizer
learning_rate = 0.0001
optimizer = torch.optim.SGD([W1,W2,b], lr=learning_rate)

# Step 3: Training Loop
for i in range(1000):

	# Model
    yhat = W1*x*x + W2*x + b

    # Loss Function
    loss = (yhat - y).pow(2).sum()
    
    # Compute gradients and update parameters
    optimizer.zero_grad()   
    loss.backward()
    optimizer.step()

W1 = W1.data.numpy()
W2 = W2.data.numpy()
b = b.data.numpy()

# Evaluation
plt.plot(x.data.numpy(),y.data.numpy(),'o')
plt.plot(x.data.numpy(),W1*x.data.numpy()*x.data.numpy()+W2*x.data.numpy()+b,'r')
plt.show()

