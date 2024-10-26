import gym
import datetime

from gym.wrappers.flatten_observation import FlattenObservation

from commons.algorithms.dql import DQLearning
from commons.network.lineal import LinealNetwork2
from commons.util import train, play, random

def runTrain(episodies, render=False):
    directory = "taxi/" + datetime.datetime.now().strftime('%Y%m%d')
    env = gym.make('Taxi-v3', render_mode='human' if render else None)
    env = FlattenObservation(env)
    network = LinealNetwork2(env.observation_space.shape[0], env.observation_space.shape[0], env.action_space.n)
    dql = DQLearning(
        network = network,
        lr = 0.001, 
        gamma = 0.99, 
        epsilon_max = 1, 
        epsilon_min = 0.02, 
        epsilon_decay = 0.000025,
        memory_length = 2000,
        mini_batch_size = 32,
        directory = directory + "/models"
    )

    train(episodies, env, dql, directory, wonEpisode, finishedEpisode, updateReward)


def runPlay():
    directory = "taxi/" + datetime.datetime.now().strftime('%Y%m%d')
    env = gym.make('Taxi-v3', render_mode='human')
    env = FlattenObservation(env)
    network = LinealNetwork2(env.observation_space.shape[0], env.observation_space.shape[0], env.action_space.n)
    dql = DQLearning(
        network = network,
        lr = 0.001, 
        gamma = 0.9, 
        epsilon_max = 0, 
        epsilon_min = 0, 
        epsilon_decay = 0,
        memory_length = 1000,
        mini_batch_size = 32,
        directory = directory + "/models"
    )

    play(env, dql, wonEpisode, finishedEpisode)

def runSample():
    env = gym.make('Taxi-v3', render_mode='human')

    random(env)

def wonEpisode(new_state,reward,terminated,truncated,info):
    return terminated and reward == 20

total_steps = 0
def finishedEpisode(new_state,reward,terminated,truncated,info):
    global total_steps

    total_steps += 1

    if total_steps >= 200 or terminated : 
        total_steps = 0
        return True
    else :
        return False

def updateReward(new_state,reward,terminated,truncated,info) : 
    return reward

if __name__ == '__main__':
    #runSample()
    #runTrain(3000, render=False)
    runPlay()