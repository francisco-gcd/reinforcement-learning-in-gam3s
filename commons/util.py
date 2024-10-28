import numpy as np
import cv2

from gym import Env

from PIL import Image
from pathlib import Path

from commons.algorithms.algorithms import Algorithms
from commons.logger import Logger

def train(episodies: int, steps: int, env: Env, alg: Algorithms, directory: str, evaluations:int, updateReward: callable):
    logger = Logger(directory)
    best_score = -float('inf')

    for e in range(episodies):
        state = env.reset()
        step = 0
        episody_reward = 0
        done = False

        while(not done and step <= steps):
            action = alg.action(state)
            new_state, reward, done, info = env.step(action)
            new_reward = updateReward(step,new_state, reward, done, info)

            alg.learn(state, action, new_state, new_reward, done)

            episody_reward += new_reward
            state = new_state
            step += 1

        logger.report_train(alg, e, episody_reward)
        alg.next_episody()

        if (e + 1) % 10 == 0:
            avg_reward = evaluate(steps, env, alg, evaluations, updateReward)
            logger.report_evaluation(e, avg_reward, best_score)
            if avg_reward >= best_score:
                best_score = avg_reward
                alg.save("partial")

    alg.save("final")

    env.close()
    logger.closeWritter()

def evaluate(steps: int, env: Env, alg: Algorithms, episodies, updateReward: callable):
    epsilon = alg.epsilon

    alg.epsilon = 0
    total_reward = 0
    for e in range(episodies):
        state = env.reset()
        episody_reward = 0
        done = False

        while (not done):
            action = alg.action(state)
            next_state, reward, done, info = env.step(action)
            new_reward = updateReward(-1, next_state, reward, done, info)

            episody_reward += new_reward
            state = next_state

        total_reward += episody_reward
#        print(f"Evaluación episodio {e} con recompensa de {episody_reward}")
    alg.epsilon = epsilon

    return total_reward / episodies

def play(env:Env, alg:Algorithms, screen_size=None):
    state = env.reset()
    done = False

    if screen_size:
        print(screen_size)
        out = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, screen_size)

    while(not done):
        action = alg.action(state)
        state,reward,done,info = env.step(action)

        frame = env.render()
        if screen_size:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if screen_size:
        out.release()

    env.close()

def sample(env:Env):
    done = False

    state = env.reset()
    while(not done):
        action = env.action_space.sample()
        state,_,done,_ = env.step(action)
        env.render()

def show_image(image):
    frames = image
    #frames = frames * 255

    im = Image.fromarray(frames)
    im.show()

def save_image(image, directory, title):
    save_dir = Path(f"{directory}")
    save_dir.mkdir(parents=True, exist_ok=True)

    frames = image

    if frames.shape[0] == 4:
        frames = np.concatenate(image[:], axis=0)

    if frames.max() <= 1:
        frames = frames * 255
        frames = frames.astype(np.uint8)

    im = Image.fromarray(frames)
    im.save(f"{directory}/{title}.jpeg")    