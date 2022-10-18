import torch
import torch.nn as nn
import random 
import numpy as np

input_dim = 1
hidden_dim = 1
num_rnn_layers = 1
seq_len = 4
batch_size = 1

torch.manual_seed(0)
rnn = nn.GRU(input_dim, hidden_dim, num_rnn_layers)

torch.manual_seed(0)
rnn_cell = nn.GRUCell(input_dim, hidden_dim)

sequence = torch.randn([seq_len, batch_size, input_dim])
print(sequence[0])
print(sequence[0].shape)

print('-'*100)
print('GRU')
hidden = torch.zeros(num_rnn_layers, batch_size, hidden_dim)
print(hidden)
gru_out, gru_hidden = rnn(sequence, hidden)
print(f"out shape: {gru_out.shape}")
print(f"hidden: {gru_hidden}")
print(f"hidden shape: {gru_hidden.shape}")
print()
    

print('-'*100)
print('GRU Cell')

hidden_cell = torch.zeros(batch_size, hidden_dim)
grucell_out = []
hidden = hidden_cell
print(hidden)
for elem in sequence:
    print(elem)
    #hidden = rnn(elem, hidden)
    hidden = rnn_cell(elem, hidden)
    grucell_out.append(hidden.item())
    print(f"hidden: {hidden}")
    print(f"hidden shape: {hidden.shape}")
    print()

print('-'*100)
print(f"GRU output: {gru_out}")
print(f"GRUCell output: {grucell_out}")

