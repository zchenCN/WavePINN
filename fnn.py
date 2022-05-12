"""
Fully connected neural networks 

@data: 2022-05-11
"""
import torch 
import torch.nn as nn

class FNN(nn.Module):
    """
    Fully connected neural networks
    """
    def __init__(self, layer_sizes, activation='tanh'):
        super(FNN, self).__init__()
        self.layer_sizes = layer_sizes 

        if activation == 'tanh':
            self.sigma = torch.tanh
        elif activation == 'sin':
            self.sigma = torch.sin 
        else:
            raise NotImplementedError(f'activation {activation} is not implemented till now')

        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            linear = nn.Linear(layer_sizes[i-1], layer_sizes[i])
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.linears.append(linear)
    
    def forward(self, x):
        for linear in self.linears[:-1]:
            x = linear(x)
            x = self.sigma(x)
        x = self.linears[-1](x)
        return x