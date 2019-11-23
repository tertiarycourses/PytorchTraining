import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,10)
        self.fc2 = nn.Linear(10,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = torch.load('nn.pkl')

x = torch.Tensor([8])
yhat = model(x)

print(yhat.data.numpy())