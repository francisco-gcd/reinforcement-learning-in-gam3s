import gym
import datetime
import numpy as np

from gym.wrappers.flatten_observation import FlattenObservation

from commons.algorithms.dql import DQLearning
from commons.network.lineal import LinealNetwork2
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

    directory="games/frozen lake/experimento 2.2/{0}/" + str(datetime.datetime.now().strftime('%Y%m%d%H%M'))
    env = gym.make('FrozenLake-v1', desc=map, is_slippery=False)
    env = FlattenObservation(env)
    network = LinealNetwork2(env.observation_space.shape[0], env.observation_space.shape[0], env.action_space.n)
    dql = DQLearning(
        directory = directory,
        network = network,
        network_updated= 5000,
        lr = 0.0005, 
        gamma = 0.9, 
        epsilon_updated = 3000,
        epsilon_max = 1, 
        epsilon_min = 0.02, 
        epsilon_decay = 0.01,
        memory_length = 64000,
        mini_batch_size = 128
    )

    train(episodies, steps, env, dql, directory, 1, updateReward)

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

    directory="games/frozen lake/experimento 2.2/{0}/" + subfolder
    env = gym.make('FrozenLake-v1', desc=map, is_slippery=False)
    env = FlattenObservation(env)
    network = LinealNetwork2(env.observation_space.shape[0], env.observation_space.shape[0], env.action_space.n)
    dql = DQLearning(
        directory = directory,
        network = network
    )

    dql.load(step)
    play(env, dql)

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
    env = FlattenObservation(env)

    sample(env)

max_steps = 100
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
    runPlay("202410202139", 484618)