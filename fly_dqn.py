import random
from collections import deque

import gym
import numpy as np
import tensorflow.keras.models
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

import datetime


class DQN:
    def __init__(self):
        # tuning params
        self.gamma = 0.9

        self.epsilon = 1
        self.epsilon_decay = 0.996
        self.epsilon_min = 0.01

        self.state_space = 8
        self.action_space = 4
        self.batch_size = 128

        self.main_network = self.build_network()
        self.target_network = self.build_network()

        self.replay_buffer = deque(maxlen=100000)

    def add_sars(self, state, action, reward, state_new, done):
        self.replay_buffer.append((state, action, reward, state_new, done))

    def get_action(self, state):
        if random.uniform(0,1) < self.epsilon:
            return np.random.randint(self.action_space)

        return self.get_action_from_model(state)

    def get_action_from_model(self, state):
        state = np.reshape(state, (1, 8))
        qvalues = self.main_network.predict(state)
        return np.argmax(qvalues[0])

    def build_network(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_space, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):

        # wait until replay buffer fill level is sufficient
        if len(self.replay_buffer) <= self.batch_size:
            return

        replay_sample = random.sample(self.replay_buffer, self.batch_size)

        x_train = np.zeros(shape=(self.batch_size, self.state_space))
        y_train = np.zeros(shape=(self.batch_size, self.action_space))

        sample = 0

        b_state = np.zeros(shape=(self.batch_size, self.state_space))
        b_state_new = np.zeros(shape=(self.batch_size, self.state_space))
        for state, action, reward, state_new, done in replay_sample:
            b_state[sample] = state
            b_state_new[sample] = state_new
            sample += 1

        action_batch = self.target_network.predict(b_state_new)
        q_batch = self.target_network.predict(b_state)

        sample = 0
        for state, action, reward, state_new, done in replay_sample:

            if done:
                target_q = reward
            else:
                state_new = np.reshape(state_new, (1, 8))
                actions = action_batch[sample] # self.target_network.predict(state_new)
                target_q = reward + self.gamma * np.amax(actions)

            state = np.reshape(state, (1, 8))
            actual_q = q_batch[sample]
            actual_q = np.reshape(actual_q, (1, 4))
            # actual_q = self.main_network.predict(state)
            actual_q[0][action] = target_q

            x_train[sample]=state
            y_train[sample]=actual_q

            sample += 1

            # self.main_network.fit(state, actual_q, batch_size=128, epochs=1, verbose=0)
        x_train.reshape(self.batch_size, 8)
        y_train.reshape(self.batch_size, 4)

        history = self.main_network.fit(x_train, y_train, epochs=1, verbose=0)
        # history.history

    def load(self, filename, episode=-1):
        path = filename + '.' + str(episode)
        self.main_network = tensorflow.keras.models.load_model(path)
        self.target_network = self.build_network()
        # self.update_target_network()

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def save(self, filename, episode=-1):
        path = filename + '.' + str(episode)
        self.target_network.save(path, overwrite=True)

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

        for steps in range(50000):

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

#######################################
## CODE

env = gym.make("LunarLander-v2")

visible_episodes = 50
record = False

p_file = 'C:/Users/Public/Documents/dev/lunar/128_128_g099_e1/model_large'

lunar_dqn = DQN()
lunar_dqn.load(p_file, 100)

fly(env, lunar_dqn, visible_episodes, record)