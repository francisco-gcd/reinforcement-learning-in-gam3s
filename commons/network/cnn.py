import torch
import numpy as np

from torch import nn

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def calculate_convolution_output(sequential: nn.Sequential, shape):
    o = sequential(torch.zeros(1, *shape))
    return int(np.prod(o.size()))

class CNNetwork1(nn.Module):
    def __init__(self, shape, action_space):
        super().__init__()

        print(f"CNN Network con {shape[0]} canales, {shape[1]} de ancho, {shape[2]} de alto y un espacio de observaciones de {action_space} acciones")

        self.action_space = action_space

        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.full_connected = nn.Sequential(
            nn.Linear(calculate_convolution_output(self.convolutional, shape), 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self.apply(initialize_weights)

    def forward(self, input):
        x = input
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.full_connected(x)

        return x

class CNNetwork2(nn.Module):
    def __init__(self, shape, action_space):
        super().__init__()

        print(f"Deep CNN Network con {shape[0]} canales, {shape[1]} de ancho, {shape[2]} de alto y un espacio de observaciones de {action_space} acciones")

        self.action_space = action_space

        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.full_connected = nn.Sequential(
            nn.Linear(calculate_convolution_output(self.convolutional, shape), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self.apply(initialize_weights)

    def forward(self, input):
        x = input
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.full_connected(x)

        return x

class CNNetwork3(nn.Module):
    def __init__(self, shape, action_space):
        super(CNNetwork3, self).__init__()

        print(f"Deep CNN Network con {shape[0]} canales, {shape[1]} de ancho, {shape[2]} de alto y un espacio de observaciones de {action_space} acciones")

        self.action_space = action_space

        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(calculate_convolution_output(self.convolutional, shape), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self.apply(initialize_weights)

    def forward(self, x):
        conv_out = self.convolutional(x).view(x.size()[0], -1)
        return self.fc(conv_out)


'''
    Para imagenes 3D
'''
class CNNetwork3D(nn.Module):
    def __init__(self, shape, action_space):
        super().__init__()

        print(f"CNN Network con {shape[0]} canales, {shape[1]} de ancho, {shape[2]} de alto y un espacio de observaciones de {action_space} acciones")

        self.action_space = action_space

        self.convolutional = nn.Sequential(
            nn.Conv3d(in_channels=shape[0], out_channels=32, kernel_size=(8,8,1), stride=4),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(8,8,1), stride=2),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,1), stride=1),
            nn.ReLU()
        )

        self.full_connected = nn.Sequential(
            nn.Linear(calculate_convolution_output(self.convolutional, shape), 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self.apply(initialize_weights)

    def forward(self, input):
        x = input
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.full_connected(x)

        return x        