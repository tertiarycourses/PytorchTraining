# Deep Learning with Pytorch
# Module 7: Natural Language Processing
# Word Embedding Demo

import torch
from torch.autograd import Variable
import torch.nn as nn

# embedding = nn.Embedding(10, 3)

# input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
# print(embedding(input))

word_to_ix = {"hello": 0, "world": 1}
embedding = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
hello_embed = embedding(Variable(torch.LongTensor([word_to_ix["hello"]])))
print(hello_embed)
