import gym
import numpy as np

class DiscretizeWrapper(gym.Wrapper):
    def __init__(self, env, interval):
        super().__init__(env)

        spaces = []
        
        for idx in range(env.observation_space.shape[0]):
            spaces.append(np.linspace(env.observation_space.low[idx], env.observation_space.high[idx], interval))

        self.spaces = np.array(spaces)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.__discretize(observation)

        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        observation = self.__discretize(observation)

        return observation
    
    def dimensions(self):

        shape = []
        for idx in range(self.spaces.ndim):
            shape.append(len(self.spaces[idx,:]))

        return shape
    
    def __discretize(self, state):
        new_state = np.zeros(state.shape[0])
        for idx in range(state.shape[0]) :
            new_state[idx] = np.digitize(state[idx], self.spaces[idx])

        return np.array(new_state, dtype=np.int64)
