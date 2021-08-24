import random
import Box2D
import gym
import numpy as np

from sarsa import Sarsa

def train(env, num_epsiodes, do_render = False):
    state = env.reset()
    success_list = np.zeros(100)
    total_reward_list = np.zeros(100)

    print("Action:", env.color_channels)
    print("Observation:", env.observation_space)

    sarsa_vlc = Sarsa()

    for episode in range(num_epsiodes):
        state = env.reset()
        a = 2
        total_reward = 0

        for steps in range(2000):

            new_state, reward, done, info = env.step(a)

            comb_state = combine_state(state)
            comb_state_new = combine_state(new_state)
            a_new = sarsa_vlc.get_action_from_Q(comb_state_new)

            # hover reward
            # reward = get_hover_reward(new_state)

            total_reward += reward

            # velocity training
            sarsa_vlc.adjust_q(comb_state, a, reward, comb_state_new, a_new)

            if do_render:
                env.render()

            if steps == 499:
                done = True

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
            a = a_new


        sarsa_vlc.calc_policy()

        if (episode % 100 == 0):
            print("Policy Size (",episode,"):", len(sarsa_vlc.policy), "/", np.prod(list(observations.values())))
            print("Success Rate:", np.average(success_list) * 100, "%")
            print("average TR:", np.average(total_reward_list))
            print("Sarsa e:", sarsa_vlc.epsilon)

    return sarsa_vlc.policy.copy()

def fly(env, num_epsiodes, record):
    success = 0
    total_reward_list = np.zeros(num_epsiodes)

    dir = 'C:/Users/Public/Documents/dev/lunar/rec'
    if record:
        env = gym.wrappers.Monitor(env, dir, force=True)

    for episode in range(num_epsiodes):
        state = env.reset()
        a = 2
        unknown = 0
        total_reward = 0

        for steps in range(50000):

            new_state, reward, done, info = env.step(a)

            comb_state_new = combine_state(new_state)
            # reward = get_hover_reward(new_state)
            try:
                a = policy[comb_state_new]
            except BaseException:
                unknown += 1
                a = 0

            env.render()
            total_reward += reward
            if done:
                if reward == 100:
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

# map the vertical velocity to discretes
def get_state(state, abs_max, steps, limit, param = "") -> int:

    f = abs_max / steps

    real_state = int(state / f)
    disc_state = int(np.min([limit, np.max([-limit, real_state])]))

    observations[param] = limit + 1

    return disc_state, real_state

def combine_state(state):
    global vlcy_real
    global vlcx_real
    global ang_real
    global xzone_real
    global yzone_real
    global av_real

    xzone, xzone_r = get_state(state[0], 1, 20, 5, "XZONE")
    yzone, yzone_r = get_state(state[1], 1.4, 5, 5, "YZONE")

    vlcx, vlcx_r = get_state(state[2], 0.7, 10, 2, "VLCX") # 0.1
    vlcy, vlcy_r = get_state(state[3], 1, 20, 5, "VLCY") # 0.1

    ang, ang_r = get_state(state[4], (np.pi / 2), 40, 10, "ANG") # 0.05
    av, av_r = get_state(state[5], 0.4, 10, 2, "AV") # 0.05
    change = False

    if (vlcy_real < vlcy_r):
        vlcy_real = vlcy_r
        change = True

    if (vlcx_real < vlcx_r):
        vlcx_real = vlcx_r
        change = True

    if (ang_real < ang_r):
        ang_real = ang_r
        change = True
    if (xzone_real < xzone_r):
        xzone_real = xzone_r
        change = True
    if (yzone_real < yzone_r):
        yzone_real = yzone_r
        change = True
    if (av_real < av_r):
        av_real = av_r
        change = True

    if change:
        print("New MAX: XY", xzone_real, yzone_real, "- VLC", vlcx_real, vlcy_real, "- ANG", ang_real, av_real)

    return (xzone, vlcy, ang)
    # return (vlc, ang, av, xzone)
    # return (xzone, yzone, vlcx, vlc, ang, av)
    # return (xzone, yzone)


#######################################
## CODE

env = gym.make("LunarLander-v2")

vlcx_real = 0
vlcy_real = 0
ang_real = 0
xzone_real = 0
yzone_real = 0
av_real = 0
observations = {}

num_epsiodes = 50000
visible_episodes = 50
record = False

dotrain = True
show_training = False

p_file = 'C:/Users/Public/Documents/dev/lunar/policy_50K_V1.json'

if dotrain:
    policy = train(env, num_epsiodes, show_training)
    save_policy(policy, p_file)
else:
    policy = load_policy(p_file)

fly(env, visible_episodes, record)