import gym
import datetime
from commons.algorithms.ql import QLearning
from commons.util import train, play, random


def runTrain(episodies, render=False):
    directory = "cliff-walking/" + datetime.datetime.now().strftime('%Y%m%d')
    env = gym.make('CliffWalking-v0', render_mode='human' if render else None)
    qlearning = QLearning(
        observation_space = env.observation_space.n,
        action_space = env.action_space.n, 
        lr = 0.9, 
        gamma = 0.9, 
        epsilon_max = 1, 
        epsilon_min = 0.02, 
        epsilon_decay = 0.0001,
        directory = directory + "/models"
    )

    train(episodies, env, qlearning, directory, gameWon)


def runPlay():
    directory = "cliff-walking/" + datetime.datetime.now().strftime('%Y%m%d')
    env = gym.make('CliffWalking-v0', render_mode='human')
    qlearning = QLearning(
        observation_space = env.observation_space.n,
        action_space = env.action_space.n, 
        lr = 0.9, 
        gamma = 0.9, 
        epsilon_max = 0, 
        epsilon_min = 0, 
        epsilon_decay = 0,
        directory = directory + "/models"
    )

    play(env, qlearning)

def runSample():
    env = gym.make('CliffWalking-v0', render_mode='human')

    random(env)

def gameWon(new_state,reward,terminated,truncated,info):
    return terminated and new_state == 47

if __name__ == '__main__':
    #runSample()
    runTrain(15000, render=False)
    runPlay()