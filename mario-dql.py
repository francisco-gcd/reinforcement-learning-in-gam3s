import datetime
import cv2

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from gym.wrappers import FrameStack, TransformObservation

from commons.wrappers.skipframe import SkipFrame
from commons.algorithms.dql import DQLearning
from commons.network.cnn import CNNetwork1, CNNetwork2, CNNetwork3
from commons.util import train, play, sample, save_image, show_image
'''
    202408240831
        network_updated= 30000,
        lr = 0.00001,
        gamma = 0.99,
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.00009,
        memory_length = 64000,
        mini_batch_size = 64

    202408252051
        network_updated= 30000,
        lr = 0.00001,
        gamma = 0.99,
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.00009,
        memory_length = 128000,
        mini_batch_size = 128
    
    202409020645
        network_updated= 30000,
        lr = 0.000001,
        gamma = 0.99,
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.00012,
        memory_length = 128000,
        mini_batch_size = 128

    202409040644
        network_updated= 30000,
        lr = 0.0001,
        gamma = 0.99,
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.00015,
        memory_length = 128000,
        mini_batch_size = 128

    202409061420        
        network_updated= 5000,
        lr = 0.00001,
        gamma = 0.99,
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.00015,
        memory_length = 128000,
        mini_batch_size = 128

    202409082142
        network_updated= 50000,
        lr = 0.00001,
        gamma = 0.99,
        epsilon_updated = 1000,
        epsilon_max = 1, 
        epsilon_min = 0.1, 
        epsilon_decay = 0.00015,
        memory_length = 128000,
        mini_batch_size = 128

        









        network_updated= 1000,
        lr = 0.00001,
        gamma = 0.99,
        epsilon_updated = 9000,
        epsilon_max = 1, 
        epsilon_min = 0.05, 
        epsilon_decay = 0.005,
        memory_length = 51200,
        mini_batch_size = 64



'''

def runTrain(episodies, steps):
    directory="games/mario/experimento 6/{0}/" + str(datetime.datetime.now().strftime('%Y%m%d%H%M'))

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, [['right'], ['right', 'A']])

    env = TransformObservation(env, f=processImage)
    env = SkipFrame(env, skip=4)
    env = FrameStack(env, num_stack=4)

    network = CNNetwork3((4, 84, 84), env.action_space.n)
    dql = DQLearning(
        directory = directory,
        network = network,  
        network_updated= 1000,
        lr = 0.00001,
        gamma = 0.99,
        epsilon_updated = 9000,
        epsilon_max = 1, 
        epsilon_min = 0.05, 
        epsilon_decay = 0.005,
        memory_length = 64000,
        mini_batch_size = 64
    )

    train(episodies, steps, env, dql, directory, 1, updateReward)

def runPlay():
    directory="games/mario/experimento 6/{0}/202410160031"
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    


    env = JoypadSpace(env, [['right'], ['right', 'A']])

    env = TransformObservation(env, f=processImage)
    env = SkipFrame(env, skip=4)
    env = FrameStack(env, num_stack=4)

    network = CNNetwork3((4, 84, 84), env.action_space.n)

    dql = DQLearning(
        directory = directory,
        network = network
    )

    dql.load(3458898)
    play(env, dql)
    

def runSample():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, [['right'], ['right', 'A']])

    env = TransformObservation(env, f=processImage)
    env = SkipFrame(env, skip=4)
    env = FrameStack(env, num_stack=4)

    sample(env)

def processImage(x):
    observation = x

    # Recortamos la imagen
    observation = observation[33:240, 0:256]

    # Reescalamos la imagen 
    observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)

    # Transformamos a escala de grises
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

    # Normalizamos la matriz
    observation = observation / 255

    return observation

#max_steps = 6480  # 320 sg para pasarse el nivel
#max_steps = 3240  # 160 sg para pasarse el nivel
max_steps = 2430  # 120 sg para pasarse el nivel
#max_steps = 1620  # 80 sg para pasarse el nivel, de sobra, lo hacen los speedrunners

def updateReward(step,state,reward,done,info) :
    return reward

if __name__ == '__main__':
    #runSample()
    #runTrain(15000, max_steps)
    runPlay()
