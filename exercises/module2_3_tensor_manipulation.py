import torch

# Reshape a torch tensor
# a = torch.linspace(1,10,10).view([2,5])
# a = torch.linspace(1,10,10).view([-1,2])
# print(a)

# Unsqueeze and squeeze dimensions
# x = torch.linspace(-1, 1, 100)
# print(x.shape)
# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) 
# print(x.shape)
# a = torch.linspace(1,10,10).view([2,5])
# b = torch.unsqueeze(a,dim=0)
# b = torch.unsqueeze(a,dim=1)
# b = torch.unsqueeze(a,dim=2)
# print(b)
# c = torch.squeeze(b)
# print(c)
# print(a)
# print(b)
# print(c)

# Concatenate
# x = torch.Tensor([1,2,3])
# y = torch.cat((x,x,x))
# print(y)

# Transpose
# x = torch.Tensor([[1,2],[3,4]])
# y = torch.t(x)
# print(x)
# print(y)

# Numpy Tutorial
# a = np.array([1,3])
# print(a.shape)
# b = np.expand_dims(a,axis=0)
# b = np.expand_dims(a,axis=1)
# b = a[np.newaxis]
# print(b.shape)
# c = np.squeeze(b)
# print(c.shape)
# print(a)
# print(b)
# print(c)

