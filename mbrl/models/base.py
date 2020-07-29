'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-24 01:02:16
@LastEditTime: 2020-07-03 21:40:23
@Description:
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.distributions import Categorical

def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var

def CPU(var):
    return var.cpu().detach()

class Unsqueeze(nn.Module):
    def __init__(self, *args):
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        B, dim = x.shape
        return x.view((B, 1, dim))

class Squeeze(nn.Module):
    def __init__(self, *args):
        super(Squeeze, self).__init__()

    def forward(self, x):
        B, _, dim = x.shape
        return x.view((B, dim))

def mlp(sizes, activation, output_activation=nn.Identity, dropout=0):
    layers = []
    #padding = int((kernel_size-1)/2)
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        dropout_layer = [ nn.Dropout(p=dropout) ] if (j < len(sizes)-2 and dropout>0) else []
        #maxpool_layer = [Unsqueeze(), nn.MaxPool1d(kernel_size, stride=1, padding=padding), Squeeze()] if j < len(sizes)-3 else []
        #print("layer ",j, " dropout layer: ", dropout_layer)
        new_layer = [nn.Linear(sizes[j], sizes[j+1]), act()] + dropout_layer #+ maxpool_layer
        layers += new_layer
    return nn.Sequential(*layers)

class MLPRegression(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_sizes=(64,64), activation=nn.Tanh):
        """
            @param int - input_dim
            @param int - output_dim 
            @param list - hidden_sizes : such as [32,32,32]
        """ 
        super().__init__()
        self.net = mlp([input_dim] + list(hidden_sizes) + [output_dim], activation)

    def forward(self, x):
        """
            @param tensor - x: shape [batch, input dim]
            
            @return tensor - out : shape [batch, output dim]
        """ 
        out = self.net(x)
        return out

class MLPCategorical(nn.Module):

    def __init__(self, input_dim, output_dim=2, hidden_sizes=(256,256,256,256), activation=nn.ELU, dropout=0):
        """
            @param int - input_dim
            @param int - output_dim, default 2
            @param list - hidden_sizes : such as [32,32,32]
        """ 
        super().__init__()
        self.logits_net = mlp([input_dim] + list(hidden_sizes) + [output_dim], activation, dropout=dropout)

    def forward(self, x):
        """
            @param tensor - x: shape [batch, input dim]

            @return tensor - out : shape [batch, 2]
        """ 
        logits = self.logits_net(x)
        #out = Categorical(logits=logits)
        output = F.log_softmax(logits, dim=1)
        return output

class GRURegression(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_sizes=(256, 256), output_sizes=(100, 100), activation=nn.ELU):
        super().__init__()

        self.embedding_dim = embedding_sizes[-1]
        self.hidden_dim = output_sizes[0]
        self.embedding_net = mlp([input_dim] + list(embedding_sizes), activation)
        self.output_net = mlp( list(output_sizes) + [output_dim], activation )
        self.cell = nn.GRUCell(self.embedding_dim, self.hidden_dim)

    def initial_hidden(self, batch_size, **kwargs):
        return CUDA(torch.zeros(batch_size, self.hidden_dim, **kwargs))
        

    def forward(self, x, h=None):
        '''
        @param x: input data at a single timestep, [tensor, (batch, input_dim)]
        @param h: hidden state at last timestep, [tensor, (batch, input_dim)]
        '''
        if h is None:
            h = self.initial_hidden(x.shape[0])
        embedding = self.embedding_net(x)
        h_next = self.cell(embedding, h)
        output = self.output_net(h_next)
        return output, h_next