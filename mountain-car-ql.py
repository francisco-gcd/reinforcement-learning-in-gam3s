import gym
import datetime

from commons.algorithms.ql import QLearning
from commons.util import train, play, sample, evaluate
from commons.wrappers.discretize import DiscretizeWrapper

def runTrain(episodies, steps):
    directory = "games/mountain-car/experimento 3.2/{0}/" + str(datetime.datetime.now().strftime('%Y%m%d%H%M'))
    env = gym.make('MountainCar-v0')
    env = DiscretizeWrapper(env, 40)
    ql = QLearning(
        directory = directory, 
        observation_space = env.dimensions(),
        action_space = env.action_space.n, 
        lr = 0.02, 
        gamma = 0.9, 
        epsilon_max = 1, 
        epsilon_min = 0.02, 
        epsilon_decay = 0.00001
    )

    train(episodies, steps, env, ql, directory, 10, updateReward)


def runPlay(subfolder, step):
    directory = "games/mountain-car/experimento 3.2/{0}/" + subfolder
    env = gym.make('MountainCar-v0')
    env = DiscretizeWrapper(env, 40)
    ql = QLearning(
        directory = directory,
        observation_space = env.dimensions(),
        action_space = env.action_space.n,
    )

    ql.load(step)
    play(env, ql)

def runEvaluate(subfolder, step):
    directory = "games/mountain-car/experimento 3.2/{0}/" + subfolder
    env = gym.make('MountainCar-v0')
    env = DiscretizeWrapper(env, 40)
    ql = QLearning(
        directory = directory,
        observation_space = env.dimensions(),
        action_space = env.action_space.n,
    )

    ql.load(step)
    avg_reward = evaluate(-1, env, ql, 10, updateReward)
    print(f"Evaluación Media {avg_reward} de los 10 episodios")

def runSample():
    env = gym.make('MountainCar-v0')
    sample(env)

max_steps = 1000

def updateReward(step,state,reward,done,info) : 
    return reward

if __name__ == '__main__':
    #runSample()
    #runTrain(200000, max_steps)
    runPlay("202410261742", 32757536)
    #runEvaluate("202410261742", 32757536)
