import gym
import numpy as np

from dqn import DQN

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