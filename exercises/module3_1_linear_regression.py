# Deep Learning with Pytorch
# Module 3: Regression with Pytorch
# Linear Regression Model

import torch
from torch.autograd import Variable 
import matplotlib.pyplot as plt

# Hyper parameters
learning_rate = 0.01

# Setup 1: Setup
X_train = [1,2,3,4,5]
y_train = [0,-1.1,-1.8,-3.1,-4.5]

# Variables
X = Variable(torch.Tensor(X_train), requires_grad=False)
y = Variable(torch.Tensor(y_train), requires_grad=False)

W = Variable(torch.randn(1), requires_grad=True)
b = Variable(torch.randn(1), requires_grad=True)

# Step 2: Optimizer
optimizer = torch.optim.SGD([W,b], lr=learning_rate)

# Step 3: Traing Loop
for i in range(1000):

	# Model
    yhat = X*W + b

    # Loss Function
    loss = (yhat - y).pow(2).sum()
    
    # Compute gradients and update parameters
    optimizer.zero_grad()   
    loss.backward()
    optimizer.step()

    print('step',i,'loss = %0.2f'%loss.data[0])

W = W.data.numpy()
b = b.data.numpy()

# Evaluation

plt.plot(X_train,y_train,'o')
plt.plot(X_train,W*X_train+b,'r')
plt.show()

torch.save(model,'nn.pkl')

