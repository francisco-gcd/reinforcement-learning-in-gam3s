import datetime
import numpy as np

import gym
from gym.wrappers.flatten_observation import FlattenObservation

from commons.algorithms.dql import DQLearning
from commons.network.lineal import LinealNetwork2
from commons.util import train, play, sample

'''
v0 new_reward = -0.0005
    202409161316  (-0.1, 1) 15000e 100max
        network_updated= 1000,
        lr = 0.01, 
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.0015,
        memory_length = 256000,
        mini_batch_size = 256

    202409170740  (-0.1, 1) 15000e 100max
        network_updated= 5000,      <--
        lr = 0.01, 
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.0015,
        memory_length = 256000,
        mini_batch_size = 256
        
    202409170915  (-0.1, 1) 15000e 100max
        network_updated= 10000,     <--
        lr = 0.01, 
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.0015,
        memory_length = 256000,
        mini_batch_size = 256

    202409171035  (-0.1, 1) 15000e 100max
        NT => EXITO
        network_updated= 1000,      <--
        lr = 0.00001,               <--
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.0015,
        memory_length = 256000,
        mini_batch_size = 256

    202409171201  (-0.1, 1) 15000e 100max
        NT: Se repite el caso anterior para ver si es un caso repetible. Se observa que se llena la memoria en un episodio más tarde que el caso anterior.
        Cuando se llena la memoria es cuando empieza a actuar la CNN. Se necesitaría más episodios de entrenamiento o una memoria más pequeña.
        En el peor de los casos podría darse que se llenara la memoria con partidas de 100 y que la red no hubiera empezado a entrenarse.
        network_updated= 1000,
        lr = 0.00001, 
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.0015,
        memory_length = 256000,
        mini_batch_size = 256

    202409171245  (-0.1, 1) 15000e 100max
        NT: Se repite el caso anterior para ver si es un caso repetible. 
        network_updated= 1000,
        lr = 0.00001, 
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.0015,
        memory_length = 256000,
        mini_batch_size = 256

    202409171342  (-0.1, 1) 15000e 100max
        network_updated= 1000,
        lr = 0.00001, 
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.0015,
        memory_length = 128000,     <--
        mini_batch_size = 128       <--

v1  Sin # new_reward = reward en función de actualización de recompensa
    202409171919  (-0.1, 1) 15000e 100max
        network_updated= 1000,
        lr = 0.01, 
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.002,
        memory_length = 256000,
        mini_batch_size = 256

    202409171920  (-0.1, 1) 15000e 100max
        network_updated= 5000,      <--
        lr = 0.01, 
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.002,
        memory_length = 256000,
        mini_batch_size = 256
        
    202409171921  (-0.1, 1) 15000e 100max
        network_updated= 10000,     <--
        lr = 0.01, 
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.002,
        memory_length = 256000,
        mini_batch_size = 256

    202409171946  (-0.1, 1) 15000e 100max
        network_updated= 1000,      <--
        lr = 0.00001,               <--
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.00175,
        memory_length = 256000,
        mini_batch_size = 256

    202409171948  (-0.1, 1) 15000e 100max
        network_updated= 1000,      
        lr = 0.0001,                <--
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.00175,
        memory_length = 256000,
        mini_batch_size = 256

    202409171949  (-0.1, 1) 15000e 100max
        network_updated= 1000,      
        lr = 0.001,                 <--
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.00175,
        memory_length = 256000,
        mini_batch_size = 256

    202409172014  (-0.1, 1) 15000e 100max
        network_updated= 1000,      
        lr = 0.00001,               
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.00175,
        memory_length = 128000,     <--
        mini_batch_size = 128       <--

    202409172015  (-0.1, 1) 15000e 100max
        network_updated= 1000,      
        lr = 0.00001,               
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.00175,
        memory_length = 64000,     <--
        mini_batch_size = 64       <--

    202409172016  (-0.1, 1) 15000e 100max
        network_updated= 1000,      
        lr = 0.00001,               
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.00175,
        memory_length = 32000,     <--
        mini_batch_size = 32       <--

v2  Recompesa que evita pisar por donde ya has ido








        network_updated= 5000,
        lr = 0.001, 
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.003,
        memory_length = 128000,
        mini_batch_size = 256


        network_updated= 5000,
        lr = 0.001, 
        gamma = 0.99, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.0025,
        memory_length = 128000,
        mini_batch_size = 256



'''

def runTrain(episodies, steps, render=False):
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

    directory="frozen-lake-dql/{0}/" + str(datetime.datetime.now().strftime('%Y%m%d%H%M'))
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False)
    #env = gym.make('FrozenLake-v1', desc=map, is_slippery=False, render_mode='human' if render else None)
    env = FlattenObservation(env)
    network = LinealNetwork2(env.observation_space.shape[0], env.observation_space.shape[0], env.action_space.n)
    dql = DQLearning(
        directory = directory,
        network = network,
        network_updated= 1000,
        lr = 0.00001, 
        gamma = 0.9, 
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.02, 
        epsilon_decay = 0.01,
        memory_length = 32000,
        mini_batch_size = 32
    )

    train(episodies, steps, env, dql, directory, updateReward)


def runPlay():
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

    directory="frozen-lake-dql/{0}/202410031718"
    env = gym.make('FrozenLake-v1', desc=map, is_slippery=False, render_mode='human')
    env = FlattenObservation(env)
    network = LinealNetwork2(env.observation_space.shape[0], env.observation_space.shape[0], env.action_space.n)
    dql = DQLearning(
        directory = directory,
        network = network
    )

    play(env, dql, 7698)

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

    #env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False)
    env = gym.make('FrozenLake-v1', desc=map, is_slippery=False, render_mode='human')
    env = FlattenObservation(env)

    sample(env)

states = []
max_steps = 100
winned = 0
'''
def updateReward(step,state,reward,done,info) : 
    global states
    global max_steps

    new_reward = 0.1

#    new_reward = reward

    for s in states:
        if np.array_equal(s, state):
            new_reward = -0.1

    if done and reward == 0:
        new_reward = -0.5

    if done and reward == 1:
        new_reward = 1 + (max_steps - len(states)) / max_steps

    if done:
        states = []
    else:
        states.append(state)

    return new_reward
'''

def updateReward(step,state,reward,done,info) : 
    global winned

    if done and reward == 0:
        return -1

    if done and reward == 1:
        winned += 1
        return 1

    return reward


if __name__ == '__main__':
    runSample()
    #runTrain(15000, max_steps, render=False)
    print(f"Winned: {winned}")
    #runPlay()