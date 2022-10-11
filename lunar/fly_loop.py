import gym
import numpy as np

from dqn import DQN

def loop(env):
    for step in range(40):
        state, reward, done, info = env.step(2)
        env.render()

    if state[5] > 0:
        dir = 1
    else:
        dir = 3

    for step in range(60):
        if step % 3 == 0:
            state, reward, done, info = env.step(dir)
        else:
            if step % 3 == 1:
                state, reward, done, info = env.step(dir)
            else:
                if step % 3 == 2:
                    state, reward, done, info = env.step(dir)
        env.render()

    # for step in range(30):
    #     state, reward, done, info = env.step(2)
    #     env.render()

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
            print("Steps:", steps)
            if steps == 100:
                print("LOOP")
                loop(env)


            new_state, reward, done, info = env.step(a)

            env.render()
            total_reward += reward
            if done:
                if reward == 100:
                    success += 1
                print("R:", reward, "TR:", total_reward, "S:", steps)
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

path_root = 'C:/Users/JAN/Documents/ri/lunar/models/'
# path_root = 'C:/Users/U429079/models/'
path_pattern = '{}/model.{}'

model = "128_128_g099_e07"
episode = 199

path = path_root + path_pattern.format(model, episode)

lunar_dqn = DQN()
lunar_dqn.load(path)

fly(env, lunar_dqn, visible_episodes, record)