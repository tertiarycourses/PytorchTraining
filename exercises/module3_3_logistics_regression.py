# Deep Learning with Pytorch
# Module 3: Regression with Pytorch
# Logistic Regression Model

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn


# Step 1: Setup
torch.manual_seed(1)    

# Hyper Parameters
EPOCH = 2               
BATCH_SIZE = 50
LR = 0.001              

# Mnist digits dataset
transform = transforms.Compose(
    [transforms.ToTensor()])

train_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True)

test_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=False,
    download=True, 
    transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(
    test_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True)

# Model
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x = self.fc(x)
        return x

model = Model(28*28,10) 
print(model) 

# Step 3: Loss Function
loss_func = nn.CrossEntropyLoss()      
                
# Step 4: Optmizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR) 

# Step 5: Training Loop
correct = 0
total = 0
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):   
        x = Variable(x.view(-1, 28*28))      
        y = Variable(y) 

        yhat = model(x)
        loss = loss_func(yhat, y)   

        optimizer.zero_grad()            
        loss.backward()                  
        optimizer.step()                

        _, y_pred = torch.max(yhat.data, 1)
        total += y.size(0)
        correct += (y_pred == y.data).sum()
        if i % 20 == 0:
            print('Epoch/Step: ', epoch, '/',i, 
                '| train loss: %.4f' % loss.data[0], 
                '| accuracy: %.2f %%' % (100 * correct / total))

correct = 0
total = 0
for (x,y) in test_loader:
    x = Variable(x.view(-1, 28*28))           
    yhat = model(x)
    _, y_pred = torch.max(yhat.data, 1)
    total += y.size(0)
    correct += (y_pred == y).sum()
print('Test accuracy: %.2f %%' % (100 * correct / total))
