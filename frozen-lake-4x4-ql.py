import gym
import datetime

from commons.algorithms.ql import QLearning
from commons.util import train, play, sample

def runTrain(episodies, steps):
    directory="games/frozen lake/experimento 1/{0}/" + str(datetime.datetime.now().strftime('%Y%m%d%H%M'))
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
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
    directory="games/frozen lake/experimento 1/{0}/" + subfolder
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
    ql = QLearning(
        directory = directory,
        observation_space = env.observation_space.n,
        action_space = env.action_space.n,
    )

    ql.load(step)
    play(env, ql)

def runSample():
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
    sample(env)

max_steps = 100

def updateReward(step,state,reward,done,info) : 
    return reward

if __name__ == '__main__':
    #runSample()
    #runTrain(20000, max_steps)
    runPlay("202410131002", 130010)