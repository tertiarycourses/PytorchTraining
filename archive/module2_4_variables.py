# Deep Learning with Pytorch
# Module 2.2: Autograd & Variable

import torch
from torch.autograd import Variable

# Create variable for a tensor
# a = torch.Tensor([5])
# v_a = Variable(a)
# print(a)
# print(v_a)
# print(v_a.data)

# a = torch.Tensor([1,2])
# v_a = Variable(a)
# print(a)
# print(v_a)
# print(v_a.data)

# Gradient
# x = torch.Tensor([5])
# x = Variable(x,requires_grad=True)
# y = x*x
# print(x.data)
# print(y.data)

# y.backward()
# print(x.grad)

# Computational graph.
x = Variable(torch.Tensor([-2]), requires_grad=True)
y = Variable(torch.Tensor([5]), requires_grad=True)
z = Variable(torch.Tensor([-4]), requires_grad=True)
f = (x+y)*z    
#print(f)

# Compute gradients.
f.backward()
print('x gradient = ',x.grad)    
print('y gradient = ',y.grad)     
print('z gradient = ',z.grad)    




