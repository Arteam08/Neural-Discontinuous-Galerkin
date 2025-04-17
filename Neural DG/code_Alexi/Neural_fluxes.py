## we define the classes for neural fluxes 
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralFlux_2_value(nn.Module):
    """MLP wich commputes the flux from interface values only"""
    def __init__(self, layers, activation=F.relu):
        """layers is a list of iintegers wich give the size of the hidden layers"""
        super(NeuralFlux_2_value, self).__init__()
        self.in_dim= 2
        self.out_dim= 1
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.in_dim, layers[0]))
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.layers.append(nn.Linear(layers[-1], self.out_dim))
        self.activation = activation

    def forward(self, x):
        """
        x is a tensor of shape (batch, 2)
        """
        for i in range(len(self.layers)):
            x = self.activation(self.layers[i](x))
        return x
    

