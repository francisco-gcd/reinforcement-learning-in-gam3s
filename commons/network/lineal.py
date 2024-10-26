import torch
from torch import nn

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class LinealNetwork1(nn.Module):
    def __init__(self, observation_space, hidden, action_space):
        super().__init__()

        self.observation_space = observation_space
        self.hidden = hidden
        self.action_space = action_space

        self.network = nn.Sequential(
            nn.Linear(in_features = observation_space, out_features = hidden),
            nn.ReLU(),
            nn.Linear(in_features = hidden, out_features = action_space)
        )        

        self.apply(initialize_weights)

    def forward(self, input):
        return self.network(input)
    
class LinealNetwork2(nn.Module):
    def __init__(self, observation_space, hidden, action_space):
        super().__init__()

        self.observation_space = observation_space
        self.hidden = hidden
        self.action_space = action_space

        self.network = nn.Sequential(
            nn.Linear(in_features = observation_space, out_features = hidden),
            nn.ReLU(),
            nn.Linear(in_features = hidden, out_features = hidden),
            nn.ReLU(),
            nn.Linear(in_features = hidden, out_features = action_space)
        )

        self.apply(initialize_weights)
    
    def forward(self, input):
        return self.network(input)    