# Deep Learning with Pytorch
# Module 2.2 Challenge : Autograd & Variable

import torch
from torch.autograd import Variable

# Create variables for tensors
x = Variable(torch.Tensor([2]), requires_grad=True)
w = Variable(torch.Tensor([3]), requires_grad=True)
b = Variable(torch.Tensor([4]), requires_grad=True)

# Build a computational graph.
y = w * x + b    

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad)     
print(w.grad)    
print(b.grad)

# Challenge 2
# x = Variable(torch.Tensor([[2]]),requires_grad=True)
# y = x**2+5*x+2 
# y.backward()
# print(x.grad)



