# Deep Learning with Pytorch
# Module 7: Natural Language Processing
# Word Embedding Demo

import torch
from torch.autograd import Variable
import torch.nn as nn

# embedding = nn.Embedding(10, 3)

# input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
# print(embedding(input))

embedding = nn.Embedding(2, 5)  

word_to_ix = {"hello": 0, "world": 1}
hello_embed = embedding(Variable(torch.LongTensor([word_to_ix["world"]])))
print(hello_embed)
