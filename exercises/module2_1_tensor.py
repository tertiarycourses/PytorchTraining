# Deep Learning with Pytorch
# Module 2.1: Tensor & Basic Pytorch Operations

import torch
import numpy as np

# Create a Torch Vector
# a = [1, 2, 3]
# b = torch.Tensor(a)
# b = torch.FloatTensor(a)
# b = torch.DoubleTensor(a)
# b = torch.IntTensor(a)
# b = torch.LongTensor(a)
# print(b)
# print(b[0])


# Create a Torch Matrix
# a = [[1, 2, 3], [4, 5, 6]]
# b = torch.Tensor(a)
# print(b)
# print(b[0])

# Create a 3D Tensor
# a = [[[1., 2.], [3., 4.]],
#      [[5., 6.], [7., 8.]]]
# b = torch.Tensor(a)
# print(b)

# numpy and torch conversion
# a = np.arange(6).reshape(2,3)
# b = torch.from_numpy(a)
# c = b.numpy()
# print(a)
# print(b)
# print(c)

# Numpy vs Torch
# a = [1,2,3,4]
# a_np = np.sum(a)
# a_np = np.mean(a)
# print(a_np)

# b = torch.Tensor(a)
#b_t = torch.sum(b)
#b_t = torch.mean(b)
# b_t = torch.max(b)
# print(b_t)

# a = [[5,6]]
# b = [[1,2],[3,4]]
# a_np = np.array(a)
# b_np = np.array(b)
# c_np = np.matmul(a_np,b_np)
# d_np = np_a.dot(b_np)
# print(c_np)
# print(d_np)

# a_t = torch.Tensor(a)
# b_t = torch.Tensor(b)
# c_t = torch.mm(t_a,t_b)
# print(c_t)

# Torch Tensors
# a = torch.diag(torch.Tensor([1,2,3]))
# a = torch.eye(3)

# Torch Max
a = torch.Tensor([[1,0,0],[1,0,0],[0,1,0],[0,0,1]])
print(torch.max(a,1))

# Torch Linspace
# a_np = np.linspace(1,5,10)
# a_t = torch.linspace(1,5,10)
# print(a_np)
# print(a_t)


# Create uniform random numbers from 0 to 1
# a = torch.rand(5, 3)
# print(a)

# Create gaussion random numbers with mean 0 and std 1
# a = torch.randn(5, 3)
# print(a)

# Tensor operations
# a = torch.Tensor([1,1])
# b = torch.Tensor([2,2])
# print(a+b)
# print(torch.add(a, b))

# Appendix 1 - Numpy Tutorial

# Matrix Multiplication
# a = np.array([[1,1],[2,2]])
# b = np.array([3,3])
# print(np.matmul(b,a))
# print(np.matmul(a,b))
# print(a.dot(b))
# print(a.T.dot(b))
# print(b*a)
# print(a*b)







