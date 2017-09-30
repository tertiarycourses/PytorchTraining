# Deep Learning with Pytorch
# Module 6: Recurrent Neural Network
# Introduction to RNN, LSTM and GRU

import torch
import torch.nn as nn
from torch.autograd import Variable 

# RNN 

# rnn = nn.RNN(10, 20, 1) 				# input_size, hidden_size, num_layers
# input = Variable(torch.randn(5, 3, 10)) # seq_len, batch, input_size
# h0 = Variable(torch.randn(1, 3, 20)) 	# num_layers * num_directions, batch, hidden_siz
# output, hn = rnn(input, h0)			# output: seq_len, batch, hidden_size * num_directions

# print(input)
# print(h0)
# print(output)

# LSTM
# rnn = nn.LSTM(10, 20, 1)
# input = Variable(torch.randn(5, 3, 10))
# h0 = Variable(torch.randn(1, 3, 20))
# c0 = Variable(torch.randn(1, 3, 20))
# output, hn = rnn(input, (h0, c0))

# print(input)
# print(h0)
# print(c0)
# print(output)

# GRU
rnn = nn.GRU(10, 20, 1)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(1, 3, 20))
output, hn = rnn(input, h0)

# print(input)
# print(h0)
# print(output)