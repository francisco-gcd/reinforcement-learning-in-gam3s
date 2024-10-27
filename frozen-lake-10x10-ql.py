import gym
import datetime
import numpy as np

from commons.algorithms.ql import QLearning
from commons.util import train, play, sample

def runTrain(episodies, steps):
    map = [
        "FFFHHHHHFF",
        "FFFFFFFFFF",
        "FFFHHHHHFF",
        "FFFHFFFHFF",
        "FFFHFFFHFF",
        "FFFFFFFHFF",
        "FFFFFFFHFF",
        "FFFHHHHHFF",
        "FFFHGFFFFF",
        "SFFHFFFFFF",
    ]

    directory="games/frozen lake/experimento 1.4/{0}/" + str(datetime.datetime.now().strftime('%Y%m%d%H%M'))
    env = gym.make('FrozenLake-v1', desc=map, is_slippery=False)
    ql = QLearning(
        directory = directory, 
        observation_space = env.observation_space.n,
        action_space = env.action_space.n, 
        lr = 0.9, 
        gamma = 0.9, 
        epsilon_max = 1, 
        epsilon_min = 0.02, 
        epsilon_decay = 0.0001
    )

    train(episodies, steps, env, ql, directory, 1, updateReward)


def runPlay(subfolder, step):
    map = [
        "FFFHHHHHFF",
        "FFFFFFFFFF",
        "FFFHHHHHFF",
        "FFFHFFFHFF",
        "FFFHFFFHFF",
        "FFFFFFFHFF",
        "FFFFFFFHFF",
        "FFFHHHHHFF",
        "FFFHGFFFFF",
        "SFFHFFFFFF",
    ]

    directory="games/frozen lake/experimento 1.4/{0}/" + subfolder
    env = gym.make('FrozenLake-v1', desc=map, is_slippery=False)
    ql = QLearning(
        directory = directory,
        observation_space = env.observation_space.n,
        action_space = env.action_space.n,
    )

    ql.load(step)
    play(env, ql)

def runSample():
    map = [
        "FFFHHHHHFF",
        "FFFFFFFFFF",
        "FFFHHHHHFF",
        "FFFHFFFHFF",
        "FFFHFFFHFF",
        "FFFFFFFHFF",
        "FFFFFFFHFF",
        "FFFHHHHHFF",
        "FFFHGFFFFF",
        "SFFHFFFFFF",
    ]

    env = gym.make('FrozenLake-v1', desc=map, is_slippery=False)
    sample(env)

max_steps = 100

'''
def updateReward(step,state,reward,done,info) : 
    new_reward = reward

    if done and reward == 0:
        new_reward = -1

    if done and reward == 1:
        new_reward = 1

    return new_reward
'''

last_state = None
states = []
def updateReward(step,state,reward,done,info) : 
    global states
    global last_state
    global max_steps

    new_reward = reward

    found = False
    for s in states:
        if np.array_equal(s, state):
            found = True
            break
    
    if not found:
        states.append(state)
        new_reward = 0.1

    if np.array_equal(last_state, state):
        new_reward = -0.1

    if done and reward == 0:
        new_reward = -0.5

    if done and reward == 1:
        new_reward = (max_steps - len(states)) / 10

    if done:
        states = [] 
        last_state = None

    last_state = state

    return new_reward

if __name__ == '__main__':
    #runSample()
    #runTrain(20000, max_steps)
    runPlay("202410162328", 599696)