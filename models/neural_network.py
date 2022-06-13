import os
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F


import numpy as np


def conv3x3(i_channels, out_channels, stride=1):
    # 3x3 convolution
    return nn.Conv2d(i_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ConvBlock(nn.Module):
    """
    A simple convolution block with Batch Normalization and ReLU activation
    """

    def __init__(self, in_channels, out_channels,stride=1):
        super(ConvBlock, self).__init__()
        self.conv = conv3x3(in_channels,out_channels,stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.conv(x)
        out = self.bn(x)
        out = self.relu(x)
        return out

class ConvolutionNetwork(nn.Module):

    def __init__(self, num_layers, num_channels, n, action_size):
        super(ConvolutionNetwork, self).__init__()

        # Convolutional architecture
        block_list = [ConvBlock(3,num_channels)] + [ConvBlock(num_channels,num_channels) for _ in range(num_layers)]
        self.conv_layers = nn.Sequential(*block_list)

        # policy head
        self.p_lin = nn.Linear(n ** 2, action_size)
        self.softmax = nn.Softmax(dim=1)

        # value Head
        self.v_fc = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self,x):

        out = self.conv_layers(x)

        # policy head
        p = self.p_lin(out)
        p = self.softmax(p)

        # value head
        v = self.v_fc(out)
        v = self.tanh(v)

        return p,v


class AlphaZeroLoss(nn.Module):

    def __init__(self):
        super(AlphaZeroLoss, self).__init__()

    def forward(self, ps, vs, target_ps, target_vs):
        value_loss = torch.mean(torch.pow(vs - target_vs, 2))
        policy_loss = -torch.mean(torch.sum(target_ps * ps, 1))

        return value_loss + policy_loss


class NetworkWrapper():

    def __init__(self, lr, l2, n_layers, n_channels, train_use_gpu=True, libtorch_use_gpu=True):

        self.num_channels = n_channels
        self.num_layers = n_layers

        self.lr = lr
        self.l2 = l2
        self.n = 3

        self.train_use_gpu = train_use_gpu
        self.libtorch_use_gpu = libtorch_use_gpu

        self.neural_network = ConvolutionNetwork(self.num_layers,self.num_channels,self.n,self.n ** 2)
        if self.train_use_gpu:
            self.neural_network.cuda()

        self.optim = Adam(self.neural_network.parameters(), lr=self.lr, weight_decay=self.l2)
        self.loss = AlphaZeroLoss()

    def train(self, examples, batch_size, epochs):
        for epoch in range(0, epochs):
            self.neural_network.train()

            train_data = random.sample(examples, batch_size)

            t_boards, t_actions, t_values = zip(*train_data)


            b_batch = torch.Tensor(t_boards)
            p_batch = torch.Tensor(t_actions)
            v_batch = torch.Tensor(t_values)

            if self.train_use_gpu:
                b_batch = b_batch.cuda()
                p_batch = p_batch.cuda()
                v_batch = v_batch.cuda()

            self.optim.zero_grad()

            ps, vs = self.neural_network(b_batch)
            loss = self.loss(ps,vs,p_batch,v_batch)
            loss.backwards()

            self.optim.step()

    def predict(self, state):
        self.neural_network.eval()

        policy, value = self.neural_network(state)

        return policy.cpu().detach().numpy() , value.cpu().detach().numpy()

    def load_model(self, folder='pretrained', filename='checkpoint'):

        filepath = os.path.join(folder,filename)
        state = torch.load(filepath)
        self.neural_network.load_state_dict(state['network'])
        self.optim.load_state_dict(state['optim'])

    def save_model(self, folder='pretrained', filename='checkpoint'):

        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder,filename)
        state = {'network' : self.neural_network.state_dict() , 'optim' : self.optim.state_dict()}
        torch.save(state,filepath)

        # save torchscript
        filepath += '.pt'
        self.neural_network.eval()

        if self.libtorch_use_gpu:
            self.neural_network.cuda()
            example = torch.rand(1, 2, self.n, self.n).cuda()
        else:
            self.neural_network.cpu()
            example = torch.rand(1, 2, self.n, self.n).cpu()

        traced_script_module = torch.jit.trace(self.neural_network, example)
        traced_script_module.save(filepath)

        if self.train_use_gpu:
            self.neural_network.cuda()
        else:
            self.neural_network.cpu()



