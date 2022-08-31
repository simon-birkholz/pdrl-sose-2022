"""
This file is used to define the neural networks.
"""

import os
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

try:
    import wandb

    USE_WANDB = True
except ImportError:
    USE_WANDB = False


def conv3x3(i_channels, out_channels, stride=1):
    # 3x3 convolution
    return nn.Conv2d(i_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def avg_pool(kernel_size=2, stride=2):
    return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0)


class ConvBlock(nn.Module):
    """
    A simple convolution block with Batch Normalization and ReLU activation
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = conv3x3(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ConvolutionNetwork(nn.Module):

    def __init__(self, num_layers, num_channels, n, action_size):
        super(ConvolutionNetwork, self).__init__()

        # Convolutional architecture
        block_list = [ConvBlock(2, num_channels)] + [ConvBlock(num_channels, num_channels) for _ in range(num_layers)]
        self.conv_layers = nn.Sequential(*block_list)

        # policy head
        self.p_conv = nn.Conv2d(num_channels, 1, kernel_size=1, padding=0, bias=False)
        self.p_bn = nn.BatchNorm2d(num_features=1)
        self.p_relu = nn.ReLU(inplace=True)

        self.p_lin = nn.Linear(n ** 2, action_size)
        self.softmax = nn.Softmax(dim=0)

        # value Head
        self.v_lin = nn.Linear(n ** 2, 256)
        self.v_fc = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv_layers(x)

        # policy head
        out = self.p_conv(out)
        out = self.p_bn(out)
        out = self.p_relu(out)

        out = torch.flatten(out, start_dim=1)

        p = self.p_lin(out)
        p = self.softmax(p)

        # value head
        v = self.v_lin(out)
        v = self.v_fc(v)
        v = self.tanh(v)

        return p, v


# TODO differences with post- and pre-activation

class ResidualBlock(nn.Module):
    """
    A residual block as described in "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    """
    # Residual block
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet build upon residual blocks.
    """
    def __init__(self, num_layers, num_channels, n, action_size):
        super(ResNet, self).__init__()

        # residual block
        res_list = [ConvBlock(2, num_channels)] + [ResidualBlock(num_channels, num_channels) for _ in range(num_layers)]
        self.res_layers = nn.Sequential(*res_list)

        # policy head
        self.p_conv = nn.Conv2d(num_channels, 1, kernel_size=1, padding=0, bias=False)
        self.p_bn = nn.BatchNorm2d(num_features=1)
        self.p_relu = nn.ReLU(inplace=True)

        self.p_lin = nn.Linear(n ** 2, action_size)
        self.softmax = nn.Softmax(dim=0)

        # value Head
        self.v_lin = nn.Linear(n ** 2, 256)
        self.v_fc = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        # residual block
        out = self.res_layers(inputs)

        # policy head
        out = self.p_conv(out)
        out = self.p_bn(out)
        out = self.p_relu(out)

        out = torch.flatten(out, start_dim=1)

        p = self.p_lin(out)
        p = self.softmax(p)

        # value head
        v = self.v_lin(out)
        v = self.v_fc(v)
        v = self.tanh(v)

        return p, v


class DenseBlock(nn.Module):
    """
    A dense block consisting of 5 convolutional layers with Batch Normalization and ReLU activation.
    Described in "Densely Connected Convolutional Networks" by Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    """

    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv3x3(out_channels * 2, out_channels)
        self.conv4 = conv3x3(out_channels * 3, out_channels)
        self.conv5 = conv3x3(out_channels * 4, out_channels)

    def forward(self, x):
        out = self.bn(x)
        conv1 = self.relu(self.conv1(out))

        conv2 = self.relu(self.conv2(conv1))
        out = self.relu(torch.cat([conv1, conv2], 1))

        conv3 = self.relu(self.conv3(out))
        out = self.relu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.relu(self.conv4(out))
        out = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.relu(self.conv5(out))
        # out = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return conv5


def _make_dense_block(block, in_channels):
    layers = [block(in_channels)]
    return nn.Sequential(*layers)


# TODO different conv parameters
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv3x3(in_channels, out_channels)

        self.avg_pool = avg_pool()

    def forward(self, x):
        out = self.bn(self.relu(self.conv(x)))
        out = self.avg_pool(out)

        return out


def _make_transition_layer(layer, in_channels, out_channels):
    modules = [layer(in_channels, out_channels)]
    return nn.Sequential(*modules)


# TODO different block and layer parameters, also: forward method probably faulty
class DenseNetwork(nn.Module):
    def __init__(self, num_layers, num_channels, n, action_size):
        super(DenseNetwork, self).__init__()

        self.conv = conv3x3(2, num_channels)
        self.relu = nn.ReLU(inplace=True)

        # residual block
        dense_list = [DenseBlock(num_channels, num_channels) for _ in range(num_layers)]
        self.dense_layers = nn.Sequential(*dense_list)

        # policy head
        self.p_conv = nn.Conv2d(num_channels, 1, kernel_size=1, padding=0, bias=False)
        self.p_bn = nn.BatchNorm2d(num_features=1)
        self.p_relu = nn.ReLU(inplace=True)

        self.p_lin = nn.Linear(n ** 2, action_size)
        self.softmax = nn.Softmax(dim=0)

        # value Head
        self.v_lin = nn.Linear(n ** 2, 256)
        self.v_fc = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.conv(x))

        out = self.dense_layers(out)

        # policy head
        out = self.p_conv(out)
        out = self.p_bn(out)
        out = self.p_relu(out)

        out = torch.flatten(out, start_dim=1)

        p = self.p_lin(out)
        p = self.softmax(p)

        # value head
        v = self.v_lin(out)
        v = self.v_fc(v)
        v = self.tanh(v)

        return p, v


class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class SEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 *, reduction=16):
        super(SEResBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SE_Block(out_channels, reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class SENet(nn.Module):
    def __init__(self, num_layers, num_channels, n, action_size):
        super(SENet, self).__init__()

        # se residual block
        res_list = [SEResBlock(2, num_channels)] + [SEResBlock(num_channels, num_channels) for _ in range(num_layers)]
        self.res_layers = nn.Sequential(*res_list)

        # policy head
        self.p_conv = nn.Conv2d(num_channels, 1, kernel_size=1, padding=0, bias=False)
        self.p_bn = nn.BatchNorm2d(num_features=1)
        self.p_relu = nn.ReLU(inplace=True)

        self.p_lin = nn.Linear(n ** 2, action_size)
        self.softmax = nn.Softmax(dim=0)

        # value Head
        self.v_lin = nn.Linear(n ** 2, 256)
        self.v_fc = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        # residual block
        out = self.res_layers(inputs)

        # policy head
        out = self.p_conv(out)
        out = self.p_bn(out)
        out = self.p_relu(out)

        out = torch.flatten(out, start_dim=1)

        p = self.p_lin(out)
        p = self.softmax(p)

        # value head
        v = self.v_lin(out)
        v = self.v_fc(v)
        v = self.tanh(v)

        return p, v


class AlphaZeroLoss(nn.Module):

    def __init__(self):
        super(AlphaZeroLoss, self).__init__()

    def forward(self, ps, vs, target_ps, target_vs):
        value_loss = torch.mean(torch.pow(vs - target_vs, 2))
        policy_loss = -torch.mean(torch.sum(target_ps * ps, 1))

        if USE_WANDB:
            wandb.log({'value_loss': value_loss, 'policy_loss': policy_loss, 'loss': value_loss + policy_loss})

        return value_loss + policy_loss


class NetworkWrapper():

    def __init__(self, n, lr, l2, n_layers, n_channels, arch='conv', train_use_gpu=False, libtorch_use_gpu=False,
                 **kwargs):

        self.num_channels = n_channels
        self.num_layers = n_layers

        self.lr = lr
        self.l2 = l2
        self.n = n

        self.train_use_gpu = train_use_gpu
        self.libtorch_use_gpu = libtorch_use_gpu

        if arch == 'conv':
            self.neural_network = ConvolutionNetwork(self.num_layers, self.num_channels, self.n, self.n ** 2)
        elif arch == 'resnet':
            self.neural_network = ResNet(self.num_layers, self.num_channels, self.n, self.n ** 2)
        elif arch == 'densenet':
            self.neural_network = DenseNetwork(self.num_layers, self.num_channels, self.n, self.n ** 2)
        elif arch == 'senet':
            self.neural_network = DenseNetwork(self.num_layers, self.num_channels, self.n, self.n ** 2)
        else:
            raise ValueError("Unknown architecture")

        if self.train_use_gpu:
            self.neural_network.cuda()

        self.optim = Adam(self.neural_network.parameters(), lr=self.lr, weight_decay=self.l2)
        self.loss = AlphaZeroLoss()

    def train(self, examples, batch_size, epochs):
        for epoch in range(0, epochs):
            self.neural_network.train()

            ss = min(batch_size, len(examples))
            train_data = random.sample(examples, ss)

            t_boards, t_actions, t_values = zip(*train_data)

            b_batch = torch.from_numpy(np.array(t_boards).astype(np.float32))
            p_batch = torch.from_numpy(np.array(t_actions).astype(np.float32))
            v_batch = torch.from_numpy(np.array(t_values).astype(np.float32))

            if self.train_use_gpu:
                b_batch = b_batch.cuda()
                p_batch = p_batch.cuda()
                v_batch = v_batch.cuda()

            self.optim.zero_grad()

            ps, vs = self.neural_network(b_batch)
            loss = self.loss(ps, vs, p_batch, v_batch)
            loss.backward()

            self.optim.step()

    def predict(self, state):
        self.neural_network.eval()

        data = np.array(state).astype(np.float32)
        tens = torch.from_numpy(data)
        if self.train_use_gpu:
            tens = tens.cuda()

        policy, value = self.neural_network(tens)

        return policy.cpu().detach().numpy(), value.cpu().detach().numpy()

    def load_model(self, folder='pretrained', filename='checkpoint'):

        filepath = os.path.join(folder, filename)
        state = torch.load(filepath)
        self.neural_network.load_state_dict(state['network'])
        self.optim.load_state_dict(state['optim'])

    def save_model(self, folder='pretrained', filename='checkpoint'):

        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        state = {'network': self.neural_network.state_dict(), 'optim': self.optim.state_dict()}
        torch.save(state, filepath)

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
