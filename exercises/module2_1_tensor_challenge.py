# Deep Learning with Pytorch
# Module 2.1 Challenge: Tensor & Basic Pytorch Operations

import torch

# Ex: Scalar operation
# a = torch.Tensor([3])
# b = torch.Tensor([4])
# c = torch.Tensor([5])
# print(a*b+c)

# Ex: Matrix operation
x = torch.Tensor([[1,1]])
w = torch.Tensor([[1,2],[3,4]])
b = torch.Tensor([[2,2]])
print(torch.mm(x,w)+b)