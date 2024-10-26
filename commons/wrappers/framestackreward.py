import gym
import numpy as np

from gym.spaces import Box
from collections import deque

class FrameStackWithReward(gym.Wrapper):
    def __init__(self, env, num_stack):
        super(FrameStackWithReward, self).__init__(env)
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)
        self.rewards = deque([], maxlen=num_stack)
        
        low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
        
    def reset(self):
        ob = self.env.reset()
        for _ in range(self.num_stack):
            self.frames.append(ob)
            self.rewards.append(0.0)  # Initialize rewards to zero
        return self._get_observation()

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        self.rewards.append(reward)
        if terminated or truncated:
            ob = self.env.reset()
            self.frames.append(ob)
            self.rewards.append(0.0)  # Reset rewards when done
        return self._get_observation(), sum(self.rewards), terminated, truncated, info

    def _get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.array(self.frames)