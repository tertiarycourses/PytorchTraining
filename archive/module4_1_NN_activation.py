# Deep Learning with Pytorch
# Module 4: Neural Network with Pytorch
# Activation Functions

import torch
from torch.autograd import Variable
import torch.nn.functional as F 
import matplotlib.pyplot as plt 

x = torch.linspace(-10,10,100)
x_np = x.numpy()

# Relu Activation Function
x_relu = F.relu(x).data.numpy()
plt.subplot(2,2,1)
plt.plot(x_np,x_relu)
plt.title('relu')

# Sigmoid Activation Function
x_sigmoid = F.sigmoid(x).data.numpy()
plt.subplot(2,2,2)
plt.plot(x_np,x_sigmoid)
plt.title('sigmoid')

# Softplus Activation Function
x_softplus = F.softplus(x).data.numpy()
plt.subplot(2,2,3)
plt.plot(x_np,x_softplus)
plt.title('softplus')

# Hyperbolic Tanh Activation Function
x_tanh = F.tanh(x).data.numpy()
plt.subplot(2,2,4)
plt.plot(x_np,x_tanh)
plt.title('tanh')

plt.show()

# Softmax Activation Function
# data = Variable( torch.randn(5) )
# print(data)
# print(F.softmax(data))
# print(F.softmax(data).sum()) 

# import numpy as np
# print(-np.log(0.73))
# print(-np.log(0.23))
