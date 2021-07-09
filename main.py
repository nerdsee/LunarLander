import random
import Box2D
import gym
import numpy as np

from sarsa import Sarsa

# map the vertical velocity to discretes
def get_state(state, f) -> int:
    disc_state = int(state / f)

    if disc_state > state_upper_limit:
        disc_state = state_upper_limit
    elif disc_state < state_lower_limit:
        disc_state = state_lower_limit

    return disc_state

def train(env, num_epsiodes):
    state = env.reset()

    print("Action:", env.action_space)
    print("Observation:", env.observation_space)

    sarsa_vlc = Sarsa()

    for episode in range(num_epsiodes):
        state = env.reset()
        a = 2

        for steps in range(5000):

            new_state, reward, done, info = env.step(a)

            comb_state = combine_state(state)
            comb_state_new = combine_state(new_state)

            # velocity training
            sarsa_vlc.adjust_q(comb_state, a, comb_state_new, reward)
            a = sarsa_vlc.get_action(comb_state_new)

            if done:
                break

            state = new_state

        sarsa_vlc.calc_policy()

        if (episode % 50 == 0):
            print("Velo (",episode,"):", sarsa_vlc.policy)

    return sarsa_vlc.policy.copy()

def combine_state(state):
    vlc = get_state(state[3], discretion_vlc)
    ang = get_state(state[4], discretion_ang)
    xzone = get_state(state[0], 0.05)
    av = get_state(state[5], 0.05)

    return (vlc, ang, xzone, av)

def fly(env, num_epsiodes, record):
    state = env.reset()
    success = 0

    dir = 'C:/Users/Public/Documents/dev/lunar/rec'
    if record:
        env = gym.wrappers.Monitor(env, dir, force=True)

    for episode in range(num_epsiodes):
        state = env.reset()
        a = 2

        for steps in range(5000):

            new_state, reward, done, info = env.step(a)

            comb_state_new = combine_state(new_state)

            try:
                a = policy[comb_state_new]
            except BaseException:
                a = 0

            env.render()

            if done:
                if reward == 100:
                    success += 1
                break

            state = new_state
    print("Landings: ", success, "/", num_epsiodes)

def save_policy(policy, filename):
    fh = open(filename, 'w')
    # json.dump(policy, fh)
    # ujson.dump(policy, fh)

    fh.write(str(policy))

    print("Saved policy to:", filename)

def load_policy(filename):
    fh = open(filename, 'r')
    # policy = ujson.load(fh)

    for i in fh.readlines():
        dic = i  # string
    policy = eval(dic)  # this is orignal dict with instace dict

    print("Loaded policy from:", filename)
    return policy

#######################################
## CODE

env = gym.make("LunarLander-v2")

discretion_vlc = 0.1
discretion_ang = 0.05
state_upper_limit = 5
state_lower_limit = state_upper_limit * -1
num_epsiodes = 5000
visible_episodes = 50
record = False

dotrain = False

p_file = 'C:/Users/Public/Documents/dev/lunar/policy_4_5000.json'

if dotrain:
    policy = train(env, num_epsiodes)
    save_policy(policy, p_file)
else:
    policy = load_policy(p_file)

fly(env, visible_episodes, record)