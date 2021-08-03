import random
from collections import deque

import gym
import numpy as np

import datetime

from dqn import DQN

def train(env, lunar_dqn, num_epsiodes, model_path, do_render = False):
    state = env.reset()
    success_list = np.zeros(100)
    total_reward_list = np.zeros(100)

    for episode in range(num_epsiodes):
        state = env.reset()
        a = lunar_dqn.get_action(state)
        total_reward = 0
        reward = 0
        e_start = datetime.datetime.now()

        step_count = 0

        for steps in range(1000):
            s_start = datetime.datetime.now()

            new_state, reward, done, info = env.step(a)

            lunar_dqn.add_sars(state, a, reward, new_state, done)
            total_reward += reward
            lunar_dqn.train()

            step_count += 1
            if (step_count % 10) == 0:
                lunar_dqn.update_target_network()

            if do_render:
                env.render()

            if done:
                if steps==200:
                    success_list[episode % 100] = 1
                    print("SUCCESS (",episode,")", total_reward)
                else:
                    success_list[episode % 100] = 0
                    if do_render:
                        print("FAILURE (",episode,")", total_reward)
                total_reward_list[episode % 100] = total_reward
                break

            state = new_state
            a = lunar_dqn.get_action(new_state)
            s_end = datetime.datetime.now()
            # print("Step",episode, "/", steps ,"# took time:", s_end - s_start)

        e_end = datetime.datetime.now()

        lunar_dqn.reduce_epsilon()
        lunar_dqn.save(model_path, episode)

        if (episode % 1 == 0):
            ave = np.minimum(episode + 1, 100)
            print("Episode (",episode,") - Success Rate:", np.sum(success_list) / ave * 100, "%", end=" - ")
            print("Epsilon:", np.round(lunar_dqn.epsilon, decimals=2), end=" - ")
            print("Last TR:", total_reward, end=" - ")
            print("Steps:", steps, end=" - ")
            print("average TR:", np.round(np.sum(total_reward_list) / ave, decimals=3), end=" - ")
            print("# took time:", e_end - e_start)
    return lunar_dqn

def fly(env, lunar_dqn, num_epsiodes, record):
    success = 0
    total_reward_list = np.zeros(num_epsiodes)

    dir = 'C:/Users/Public/Documents/dev/lunar/rec'
    if record:
        env = gym.wrappers.Monitor(env, dir, force=True)

    for episode in range(num_epsiodes):
        state = env.reset()
        a = lunar_dqn.get_action_from_model(state)
        unknown = 0
        total_reward = 0

        for steps in range(500):

            new_state, reward, done, info = env.step(a)

            env.render()
            total_reward += reward
            if done:
                if steps==200:
                    success += 1
                print("R:", reward, "TR:", total_reward, "U:", unknown)
                break

            state = new_state
            a = lunar_dqn.get_action_from_model(state)

        total_reward_list[episode % num_epsiodes] = total_reward
    print("Landings: ", success, "/", num_epsiodes)
    print("Average TR: ", np.average(list(total_reward_list)))

#######################################
## CODE

env = gym.make("MountainCar-v0")
env._max_episode_steps = 1000

num_epsiodes = 5000
visible_episodes = 50
record = False

dotrain = False
show_training = False

#     def __init__(self, gamma = 0.99, epsilon = 1, epsilon_decay = 0.9, epsilon_min=0.01, first=256, second=256, batch_size=64):

# Hyperparameters
gamma = 0.99
epsilon = 1
epsilon_decay = 0.996
epsilon_min = 0.01
first = 128
second = 128
batch_size = 64
buf = 2000

path_root = 'C:/Users/JAN/Documents/ri/mountaincar/'
# path_root = 'C:/Users/U429079/models/mountaincar/'

path_pattern = '{}_{}_g{}_e{}_ed{}_b{}.buf{}/model'
path = path_root + path_pattern.format(first, second, gamma, epsilon, epsilon_decay, batch_size, buf)

lunar_dqn = DQN(gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, batch_size=batch_size, replay_buffer=buf)

lunar_dqn.set_topology(state_space = 2, first=first, second=second, action_space = 3)

if dotrain:
    # lunar_dqn.load(p_file)
    lunar_dqn = train(env, lunar_dqn, num_epsiodes, path, show_training)
    # lunar_dqn.save(p_file)
else:
    lunar_dqn.load(path, 70)
    fly(env, lunar_dqn, visible_episodes, record)