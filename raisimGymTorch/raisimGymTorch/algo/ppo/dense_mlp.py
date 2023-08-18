from torch.distributions import Normal
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule

class ResidualDenseBlock(nn.Module):
    def __init__(self, input_size, growth_rate, num_layers, activation_fn):
        super(ResidualDenseBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(input_size + i * growth_rate, growth_rate))

        self.activation_fn = activation_fn

    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            out = layer(torch.cat(outputs, dim=-1))
            out = self.activation_fn()(out)
            outputs.append(out)

        return torch.cat(outputs[1:], dim=-1) + x


class DeepMLP_network(nn.Module):
    def __init__(self, input_size, output_size, num_blocks=2):
        super(DeepMLP_network, self).__init__()

        self.activation_fn = nn.LeakyReLU
        hidden_dim = 256
        growth_rate = 64
        layers = hidden_dim // growth_rate

        self.initial_layer = nn.Linear(input_size, hidden_dim)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResidualDenseBlock(hidden_dim, growth_rate, layers, self.activation_fn))

        self.final_layer = nn.Linear(hidden_dim, output_size)

        self.init_weights(self.initial_layer)
        for block in self.blocks:
            self.init_weights(block)
        self.init_weights(self.final_layer)
        self.input_shape = [input_size]
        self.output_shape = [output_size]


    def forward(self, obs):
        x = self.activation_fn()(self.initial_layer(obs))

        for block in self.blocks:
            x = block(x)

        return self.final_layer(x)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        elif isinstance(module, ResidualDenseBlock):
            for layer in module.layers:
                torch.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
