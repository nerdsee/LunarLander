import random
from collections import deque

import gym
import keras.models
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
class DQN:
    def __init__(self, gamma = 0.99, epsilon = 1, epsilon_decay = 0.9, epsilon_min=0.01, batch_size=64, replay_buffer=10000):
        # tuning params
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_buffer)

    def set_topology(self, state_space=2, first=128, second=128, action_space=3):
        self.first = first
        self.second = second
        self.state_space = state_space
        self.action_space = action_space

        self.main_network = self.build_network()
        self.target_network = self.build_network()

    def add_sars(self, state, action, reward, state_new, done):
        self.replay_buffer.append((state, action, reward, state_new, done))

    def get_action(self, state):
        if random.uniform(0,1) < self.epsilon:
            return np.random.randint(self.action_space)

        return self.get_action_from_model(state)

    def get_action_from_model(self, state):
        state = np.reshape(state, (1, self.state_space))
        qvalues = self.main_network.predict(state)
        return np.argmax(qvalues[0])

    def build_network(self):
        model = Sequential()
        model.add(Dense(self.first, input_dim=self.state_space, activation='relu'))
        model.add(Dense(self.second, activation='relu'))
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
                state_new = np.reshape(state_new, (1, self.state_space))
                actions = action_batch[sample] # self.target_network.predict(state_new)
                target_q = reward + self.gamma * np.amax(actions)

            state = np.reshape(state, (1, self.state_space))
            actual_q = q_batch[sample]
            actual_q = np.reshape(actual_q, (1, self.action_space))
            # actual_q = self.main_network.predict(state)
            actual_q[0][action] = target_q

            x_train[sample]=state
            y_train[sample]=actual_q

            sample += 1

            # self.main_network.fit(state, actual_q, batch_size=128, epochs=1, verbose=0)
        x_train.reshape(self.batch_size, self.state_space)
        y_train.reshape(self.batch_size, self.action_space)

        history = self.main_network.fit(x_train, y_train, epochs=1, verbose=0)
        # history.history

    def load(self, filename, episode=-1):

        path = filename +  ( ( '.' + str(episode) ) if episode >= 0 else '' )
        self.main_network = keras.models.load_model(path)
        self.target_network = self.build_network()
        # self.update_target_network()

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def save(self, filename, episode=-1):
        path = filename + '.' + str(episode)
        self.target_network.save(path, overwrite=True)

