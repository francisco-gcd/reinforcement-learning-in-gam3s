import gym
import datetime

from commons.algorithms.dql import DQLearning
from commons.network.lineal import LinealNetwork2
from commons.wrappers.discretize import DiscretizeWrapper
from commons.util import train, play, sample, evaluate

def runTrain(episodies, steps):
    directory = "games/mountain-car/experimento 4/{0}/" + str(datetime.datetime.now().strftime('%Y%m%d%H%M'))
    env = gym.make('MountainCar-v0')
    env = DiscretizeWrapper(env, 10)
    network = LinealNetwork2(env.observation_space.shape[0], 32, env.action_space.n)
    dql = DQLearning(
        directory = directory,
        network = network,
        network_updated = 5000,
        lr = 0.00001, 
        gamma = 0.9, 
        epsilon_updated = 15000,
        epsilon_max = 1, 
        epsilon_min = 0.02, 
        epsilon_decay = 0.01,
        memory_length = 256000,
        mini_batch_size = 256
    )

    train(episodies, steps, env, dql, directory, 10, updateReward)


def runPlay(subfolder, step):
    directory = "games/mountain-car/experimento 4/{0}/" + subfolder
    env = gym.make('MountainCar-v0')
    env = DiscretizeWrapper(env, 10)
    network = LinealNetwork2(env.observation_space.shape[0], 32, env.action_space.n)
    dql = DQLearning(
        directory = directory,
        network = network
    )

    dql.load(step)
    play(env, dql)

def runEvaluate(subfolder, step):
    directory = "games/mountain-car/experimento 4/{0}/" + subfolder
    env = gym.make('MountainCar-v0')
    env = DiscretizeWrapper(env, 10)
    network = LinealNetwork2(env.observation_space.shape[0], 32, env.action_space.n)
    dql = DQLearning(
        directory = directory,
        network = network
    )

    dql.load(step)
    play(env, dql)

    avg_reward = evaluate(-1, env, dql, 10, updateReward)
    print(f"Evaluación Media {avg_reward} de los 10 episodios")


def runSample():
    env = gym.make('MountainCar-v0')
    sample(env)

max_steps = 1000

def updateReward(step,state,reward,done,info) : 
    return reward

if __name__ == '__main__':
    #runSample()
    runTrain(20000, max_steps)
    #runPlay("202410231703", 2009262)
    #runEvaluate("202410231703", 2009262)