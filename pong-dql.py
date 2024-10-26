import numpy as np

import datetime
import cv2

import gym
from gym.wrappers import FrameStack, TransformObservation

from commons.algorithms.dql import DQLearning
from commons.network.cnn import CNNetwork2
from commons.util import train, play, sample, save_image

def runTrain(episodies, steps):
    directory="games/pong/experimento 5/{0}/" + str(datetime.datetime.now().strftime('%Y%m%d%H%M'))

    env = gym.make('PongNoFrameskip-v4')
    env = TransformObservation(env, f=processImage)
    env = FrameStack(env, num_stack=4)

    network = CNNetwork2((4, 84, 84), env.action_space.n)
    dql = DQLearning(
        directory = directory,
        network = network,
        network_updated= 1000,
        lr = 0.00001,
        gamma = 0.99,
        epsilon_updated = 18000,
        epsilon_max = 1, 
        epsilon_min = 0.05, 
        epsilon_decay = 0.005,
        memory_length = 64000,
        mini_batch_size = 64
    )

    train(episodies, steps, env, dql, directory, 1, updateReward)

def runPlay(subfolder, step):
    directory="games/pong/experimento 5/{0}/" + subfolder
    env = gym.make('PongNoFrameskip-v4', render_mode='human')
    env = TransformObservation(env, f=processImage)
    env = FrameStack(env, num_stack=4)

    network = CNNetwork2((4, 84, 84), env.action_space.n)
    dql = DQLearning(
        directory = directory,
        network = network
    )

    dql.load(step)
    play(env, dql)

def runSample():
    env = gym.make('PongNoFrameskip-v4', render_mode='human')
    env = TransformObservation(env, f=processImage)
    env = FrameStack(env, num_stack=4)

    sample(env)

def processImage(x):
    if isinstance(x, tuple) :
        observation = np.array(x[0])

    if isinstance(x, np.ndarray) :
        observation = x

    # Recortamos la imagen
    observation = observation[34:194, 0:160]

    # Reescalamos la imagen 
    observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)

    # Transformamos a blanco y negro
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)

    return binary

player_one_points = 0
player_two_points = 0

max_steps = 1000
def updateReward(step,state,reward,done,info) : 
    return reward

if __name__ == '__main__':
    #runSample()
    #runTrain(6000, max_steps)
    runPlay('202410220649', 5125120)