import copy
import random

import torch

import numpy as np

from commons.algorithms.algorithms import Algorithms
from collections import deque
from pathlib import Path
from torch import nn

class DQLearning(Algorithms):
    def __init__(self, directory, network: nn.Module, 
            network_updated = 0, lr = 0, gamma = 0, epsilon_max = 0, epsilon_min = 0, epsilon_decay = 0, epsilon_updated = 0, memory_length = 0, mini_batch_size = 0
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Ejecutando algoritmo en {}".format(self.device))

        self.directory = directory.format("models")
        save_dir = Path(self.directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.steps = 0
        self.q_max = 0
        self.td_error = 0
        self.v_avg = 0
        self.q_values = np.array([])

        self.lr = lr
        self.gamma = gamma
        self.epsilon_updated = epsilon_updated
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_max

        self.memory = deque([], memory_length)

        self.mini_batch_size = mini_batch_size

        self.network_updated = network_updated

        self.q_network = copy.deepcopy(network)
        self.target_network = copy.deepcopy(network)

        self.q_network.load_state_dict(network.state_dict())
        self.target_network.load_state_dict(network.state_dict())

        self.q_network = self.q_network.to(self.device)
        self.target_network = self.target_network.to(self.device)

        self.q_network.train()
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), self.lr)

    def action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.q_network.action_space - 1)
        else:
            with torch.no_grad():
                array_state = np.array(state)
                tensor_state = torch.tensor(array_state, dtype = torch.float32, device = self.device).unsqueeze(0)

                q_values = self.q_network(tensor_state)
                action = q_values.max(1)[1].item()

        return action
    
    def learn(self, state, action, new_state, reward, done):
        self.memory.append((state, action, new_state, reward, done))

        if len(self.memory) > self.mini_batch_size:
            mini_batch = random.sample(self.memory, self.mini_batch_size)
            self.__optimize(mini_batch)

        if self.steps % self.epsilon_updated == 0:
            self.epsilon = self.epsilon - self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        if self.steps % self.network_updated == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.steps += 1

    def next_episody(self):
        return

    def save(self):
        torch.save(self.q_network.state_dict(), self.directory + f"/model-dql-{self.steps}.pkl")

    def load(self, step):
        state_dict = torch.load(self.directory + f"/model-dql-{step}.pkl")
        self.q_network.load_state_dict(state_dict)

    def __optimize(self, mini_batch):
        states, actions, new_states, rewards, dones = zip(*mini_batch)

        tensor_states = torch.tensor(np.array(states), dtype = torch.float32, device= self.device)
        tensor_actions = torch.tensor(actions, dtype = torch.int64, device= self.device).unsqueeze(1)
        tensor_new_states = torch.tensor(np.array(new_states), dtype = torch.float32, device= self.device)
        tensor_rewards = torch.tensor(rewards, dtype = torch.float32, device= self.device).unsqueeze(1)
        tensor_dones = torch.tensor(dones, dtype = torch.float32, device= self.device).unsqueeze(1)
        
        current_q_values = self.q_network(tensor_states).gather(1, tensor_actions)
        self.q_values = current_q_values.cpu().detach().numpy();

        next_q_values = self.target_network(tensor_new_states).max(1)[0].detach().unsqueeze(1)
        expected_q_values = tensor_rewards + (self.gamma * next_q_values * (1 - tensor_dones))

        self.td_error = (expected_q_values - current_q_values).mean().item()
        loss = nn.SmoothL1Loss()(current_q_values, expected_q_values)

        self.q_max = current_q_values.max().item()
        self.v_avg = current_q_values.mean().item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()