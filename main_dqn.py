import random
from collections import deque

import gym
import numpy as np
import tensorflow.keras.models
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import datetime
e_start = datetime.datetime.now()
e_end = datetime.datetime.now()
print(e_start - e_end)

class DQN:
    def __init__(self):
        # tuning params
        self.gamma = 0.9

        self.epsilon = 0.8
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01

        self.state_space = 8
        self.action_space = 4
        self.batch_size = 100

        self.main_network = self.build_network()
        self.target_network = self.build_network()

        self.replay_buffer = deque(maxlen=10000)

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
        model.add(Dense(12, input_dim=8, activation='relu'))
        model.add(Dense(10, activation='relu'))
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

        for state, action, reward, state_new, done in replay_sample:
            if done:
                target_q = reward
            else:
                state_new = np.reshape(state_new, (1, 8))
                actions = self.target_network.predict(state_new)
                target_q = reward + self.gamma * np.amax(actions)

            state = np.reshape(state, (1, 8))
            actual_q = self.main_network.predict(state)
            actual_q[0][action] = target_q
            self.main_network.fit(state, actual_q, epochs=1, verbose=0)

    def load(self, filename):
        self.main_network = tensorflow.keras.models.load_model(filename)
        self.target_network = self.build_network()
        # self.update_target_network()

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def save(self, filename):
        self.target_network.save(filename, overwrite=True)

def train(env, num_epsiodes, do_render = False):
    state = env.reset()
    success_list = np.zeros(100)
    total_reward_list = np.zeros(100)

    lunar_dqn = DQN()

    for episode in range(num_epsiodes):
        state = env.reset()
        a = lunar_dqn.get_action(state)
        total_reward = 0
        reward = 0
        e_start = datetime.datetime.now()

        step_count = 0

        for steps in range(2000):
            s_start = datetime.datetime.now()

            new_state, reward, done, info = env.step(a)
            lunar_dqn.add_sars(state, a, reward, new_state, done)
            total_reward += reward
            lunar_dqn.train()

            step_count += 1
            if (step_count % 100) == 0:
                lunar_dqn.update_target_network()
                lunar_dqn.reduce_epsilon()

            if do_render:
                env.render()

            if done:
                if reward == 100:
                    success_list[episode % 100] = 1
                    if do_render:
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
            print("Step",episode, "/", steps ,"# took time:", s_end - s_start)

        e_end = datetime.datetime.now()

        if (episode % 1 == 0):
            ave = np.minimum(episode + 1, 100)
            print("Success Rate(",episode,"):", np.sum(success_list) / ave * 100, "%")
            print("Epsilon:", lunar_dqn.epsilon)
            print("Last TR:", total_reward)
            print("average TR:", np.sum(total_reward_list) / ave)
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

num_epsiodes = 50
visible_episodes = 50
record = False

dotrain = True
show_training = False

p_file = 'C:/Users/Public/Documents/dev/lunar/model_V1'

if dotrain:
    lunar_dqn = train(env, num_epsiodes, show_training)
    lunar_dqn.save(p_file)
else:
    lunar_dqn = DQN()
    lunar_dqn.load(p_file)

fly(env, lunar_dqn, visible_episodes, record)