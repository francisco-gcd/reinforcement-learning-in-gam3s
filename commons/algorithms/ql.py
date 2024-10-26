import numpy as np
import pickle
import random

import collections.abc

from pathlib import Path
from commons.algorithms.algorithms import Algorithms

class QLearning(Algorithms):
    def __init__(self, directory, observation_space, action_space, 
            lr = 0, gamma = 0, epsilon_max = 0, epsilon_min = 0, epsilon_decay = 0
        ):
        self.directory = directory.format("models")
        save_dir = Path(self.directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.steps = 0
        self.q_values = np.array([])

        self.lr = lr
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_max

        self.action_space = action_space

        shape = np.hstack((observation_space, action_space))
        self.qtable = np.zeros(shape)

    def action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_space - 1)
        else:
            t_state = tuple(state if isinstance(state, (collections.abc.Sequence, np.ndarray)) else (state,))
            action = np.argmax(self.qtable[t_state])

        return action

    def learn(self, state, action, new_state, reward, done):
        t_state = tuple(np.hstack((state, action)))
        t_new_state = tuple(new_state if isinstance(new_state, (collections.abc.Sequence, np.ndarray)) else (new_state,))

        self.qtable[t_state] = self.qtable[t_state] + self.lr * (reward + self.gamma * np.max(self.qtable[t_new_state]) - self.qtable[t_state])
        self.q_values = np.array(self.qtable[t_state])
        
        self.steps += 1

    def next_episody(self):
        self.epsilon = self.epsilon - self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def save(self):
        f = open(self.directory + f"/model-ql-{self.steps}.pkl","wb")
        pickle.dump(self.qtable, f)
        f.close()

    def load(self, step):
        f = open(self.directory + f"/model-ql-{step}.pkl","rb")
        self.qtable = pickle.load(f)

        f.close()