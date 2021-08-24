import random
from collections import deque

import gym
import numpy as np

import datetime

from dqn import DQN

def fly(env, model_hover, model_fly, num_epsiodes, record):
    success = 0
    total_reward_list = np.zeros(num_epsiodes)

    dir = 'C:/Users/Public/Documents/dev/lunar/rec'
    if record:
        env = gym.wrappers.Monitor(env, dir, force=True)


    for episode in range(num_epsiodes):
        lunar_dqn = model_hover
        state = env.reset()
        a = lunar_dqn.get_action_from_model(state)
        unknown = 0
        total_reward = 0

        for steps in range(1500):

            if steps == 300:
                lunar_dqn = model_fly
                print("swap models")

            new_state, reward, done, info = env.step(a)

            env.render()
            total_reward += reward
            if done:
                if reward == 100:
                    success += 1
                print("R:", reward, "TR:", total_reward, "U:", unknown)
                break

            state = new_state
            a = lunar_dqn.get_action_from_model(state)

        total_reward_list[episode % num_epsiodes] = total_reward
    print("Landings: ", success, "/", num_epsiodes)
    print("Average TR: ", np.average(list(total_reward_list)))

def fly_model_multi(env, model, episodes, steps, render, record, episode):
    success = 0

    dir = 'C:/Users/Public/Documents/dev/lunar/rec'
    if record:
        env = gym.wrappers.Monitor(env, dir, force=True)

    total_reward = 0
    total_steps = 0

    for e in range(episodes):
        state = env.reset()
        a = model.get_action_from_model(state)
        for step in range(steps):

            new_state, reward, done, info = env.step(a)
            total_reward += reward

            if render:
                env.render()


            if done:
                total_steps += step
                break

            state = new_state
            a = model.get_action_from_model(state)
        else:
            total_steps += step

    step_average = total_steps/episodes

    print("Episode ", episode, end=' - ')
    print("Steps Average: ", step_average, end=' - ')
    print("Reward: ", total_reward)

    return state

def fly_model_single(env, state, model, steps, render, record):

    dir = 'C:/Users/Public/Documents/dev/lunar/rec'
    if record:
        env = gym.wrappers.Monitor(env, dir, force=True)

    total_reward = 0

    a = model.get_action_from_model(state)
    for step in range(steps):

        new_state, reward, done, info = env.step(a)
        total_reward += reward

        if render:
            env.render()

        if done:
            break

        state = new_state
        a = model.get_action_from_model(state)

    print("Steps: ", step, end=' - ')
    print("Reward: ", total_reward)

    return state


#######################################
## CODE

env = gym.make("LunarLander-v2")

visible_episodes = 50
record = False

dotrain = False
show_training = False

#     def __init__(self, gamma = 0.99, epsilon = 1, epsilon_decay = 0.9, epsilon_min=0.01, first=256, second=256, batch_size=64):

# Hyperparameters
gamma = 0.99
epsilon = 0.3
epsilon_decay = 0.998
epsilon_min = 0.01
first = 150
second = 120
batch_size = 64
buf = 10000

# path_root = 'C:/Users/Public/Documents/dev/lunar/'
# path = 'models/150_120_g0.99_e1_ed0.998_b64.buf10000.hover_right/model'
path = 'models/150_120_g0.99_e1_ed0.998_b64.buf10000.hover_left/model'


# retrain fly_model to recover after hover
gamma = 0.99
epsilon = 0.3
epsilon_decay = 0.998
epsilon_min = 0.01
first = 150
second = 120
batch_size = 64
buf = 10000
num_episodes = 5000

model_fly = DQN(gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                batch_size=batch_size, replay_buffer=buf)
model_fly.set_topology(state_space=8, first=150, second=120, action_space=4)

state = env.reset()

explore = False

if not explore:
    e = 536
    model_fly.load('models/150_120_g0.99_e1_ed0.998_b64.buf10000.hover_left/model', e)
    state = fly_model_single(env, state, model_fly, 200, True, record)

    print("swap")

    e = 720
    model_fly.load('models/150_120_g0.99_e1_ed0.998_b64.buf10000.hover_right/model', e)
    state = fly_model_single(env, state, model_fly, 200, True, record)

    print("land")

    e = 1
    model_fly.load('models/fly', e)
    state = fly_model_single(env, state, model_fly, 300, True, record)

if explore:
    for e in range(400, 1040):
        model_fly.load('models/150_120_g0.99_e1_ed0.998_b64.buf10000.hover_right/model', e)
        fly_model_multi(env, model_fly, 3, 1500, False, record, e)

# left
# 447, 444, 460
# 532 534 536 # gute Reihe

#right
# 719 720 722 725