import random
import Box2D
import gym
import numpy
import numpy as np

from sarsa import Sarsa

ACTIONS_NAME = ["LF","DW","RG","UP"]

def e_g(Q, state, epsilon):
    ru = np.random.uniform(0, 1)
    if ru < epsilon:
        return np.random.randint(4)
        # return env.action_space.sample()
    else:
        a = max(list(range(env.color_channels.n)), key = lambda x: Q[(state, x)])
        return a

def train(env, Q, num_epsiodes, do_render = False):
    state = env.reset()
    success_list = np.zeros(100)
    total_reward_list = np.zeros(100)
    max_success = 0

    print("Action:", env.color_channels)
    print("Observation:", env.observation_space)

    sarsa_vlc = Sarsa()

    for episode in range(num_epsiodes):
        state = env.reset()

        # a = sarsa_vlc.get_action(state)
        a = e_g(Q, state, 0.8)

        total_reward = 0

        for steps in range(1000):

            new_state, reward, done, info = env.step(a)

            # only in Q learning
            # new_a = sarsa_vlc.get_action_from_Q(new_state)

            # for sarsa learning
            new_a = sarsa_vlc.get_action_e(new_state, 0.8)
            # new_a = e_g(Q, new_state, 0.8)

            a1 = sarsa_vlc.get_action_e(new_state, 0)
            a2 = e_g(Q, new_state, 0)
            if a1 != a2:
                print("ASSERT")
                a1 = sarsa_vlc.get_action_e(new_state, 0)
                a2 = e_g(Q, new_state, 0)

            total_reward += reward

            sarsa_vlc.adjust_q(state, a, reward, new_state, new_a)

            alpha = 0.85
            gamma = 0.9
            Q[state, a] += alpha * (reward + gamma * Q[(new_state, new_a)] - Q[(state, a)])

            if do_render:
                env.render()
                print("next: ", new_a)

            state = new_state
            a = new_a

            if done:
                if reward == 1:
                    success_list[episode % 100] = 1
                    # ^print("SUCCESS", total_reward)
                else:
                    success_list[episode % 100] = 0

                total_reward_list[episode % 100] = total_reward
                break

        if (episode % 100 == 0):
            success_rate = np.average(success_list) * 100
            if success_rate > max_success:
                max_success = success_rate

            print("Policy Size (",episode,"):", len(sarsa_vlc.policy), "/", env.observation_space.n)
            print("Success Rate (cur/max):", success_rate, "%" , "/", max_success, "%")
            print("average TR:", np.average(total_reward_list))
            print("Sarsa e:", sarsa_vlc.epsilon)
            pass

    env.render()
    sarsa_vlc.calc_policy()
    local_policy = calc_policy(Q)
    print("L", [ACTIONS_NAME[local_policy[s]] for s in range(env.observation_space.n) ])
    print("S", [ACTIONS_NAME[sarsa_vlc.policy[s]] for s in range(env.observation_space.n) ])

    ret = local_policy
    # ret = sarsa_vlc.policy.copy()

    return ret

def calc_policy(Q):
    policy = np.zeros(env.observation_space.n, numpy.int8)
    for s in range(env.observation_space.n):
        options = [Q[(s, a)] for a in range(env.color_channels.n)]
        policy[s] = np.argmax(options)
    return policy

def fly(env, policy, num_epsiodes, record):
    success = 0
    total_reward_list = np.zeros(100)

    dir = 'C:/Users/Public/Documents/dev/lunar/rec'
    if record:
        env = gym.wrappers.Monitor(env, dir, force=True)

    for episode in range(num_epsiodes):
        state = env.reset()
        a = policy[state]
        unknown = 0
        total_reward = 0

        for steps in range(50000):

            new_state, reward, done, info = env.step(a)

            try:
                a = policy[new_state]
            except BaseException:
                unknown += 1
                a = 0

            env.render()
            print("Next:", ACTIONS_NAME[a])
            total_reward += reward
            if done:
                if reward == 1:
                    success += 1
                print("R:", reward, "TR:", total_reward, "U:", unknown)
                break

            state = new_state
        total_reward_list[episode % num_epsiodes] = total_reward
    print("Landings: ", success, "/", num_epsiodes)
    print("Average TR: ", np.average(list(total_reward_list)))

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

env = gym.make("FrozenLake-v0", is_slippery=False)

num_epsiodes = 50000
visible_episodes = 50
record = False

dotrain = True

p_file = 'C:/Users/Public/Documents/dev/lunar/policy_lake.json'

local_Q = {}
for s in range(env.observation_space.n):
    for a in range(env.color_channels.n):
        local_Q[(s, a)] = 0

if dotrain:
    policy = train(env, local_Q, num_epsiodes, False)
    save_policy(policy, p_file)
else:
    policy = load_policy(p_file)

fly(env, policy, visible_episodes, record)