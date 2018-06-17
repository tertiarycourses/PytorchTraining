# Deep Learning with Pytorch
# Module 7: Recurrent Neural Network
# Word Embedding Challenge

import torch
from torch.autograd import Variable
import torch.nn as nn


# Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

vocab = set(test_sentence)

word_to_ix = {word: i for i, word in enumerate(vocab)}
embedding = nn.Embedding(len(vocab), 5) 
child_embed = embedding(Variable(torch.LongTensor([word_to_ix["child"]])))
print(child_embed)
