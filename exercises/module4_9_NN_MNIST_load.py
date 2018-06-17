# Deep Learning with Pytorch
# Module 4: Neural Network with Pytorch
# Load Model and Parameters

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn

# Load test data
test_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=False,
    download=True, 
    transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(
    test_data, 
    batch_size=100, 
    shuffle=True, 
    num_workers=2)

model = torch.load('mnist.pkl')

correct = 0
total = 0

for (x,y) in test_loader:
    x = Variable(x.view(-1, 28*28))           
    yhat = model(x)
    _, y_pred = torch.max(yhat.data, 1)
    total += y.size(0)
    correct += (y_pred == y).sum()
print('Test accuracy: %.2f %%' % (100 * correct / total))





