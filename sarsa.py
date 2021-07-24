import numpy as np

class Sarsa:

    # Lunar
    alpha = 0.1 # 0.1
    gamma = 0.9 # 0.7
    epsilon = 0.2 # 0.7
    eps_min = 0.005
    eps_f = 0.999

    # Lake
    # alpha = 0.85 # 0.1
    # gamma = 0.9 # 0.7
    # epsilon = 0.8 # 0.7
    # eps_min = 0.005
    # eps_f = 1

    def adjust_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon * self.eps_f

    def __init__(self):
        self.Q = {}
        self.policy = {}
        self.states = []

    def get_q(self, state, action):
        if not (state, action) in self.Q.keys():
            self.Q[(state, action)] = 0.0

        return self.Q[(state, action)]

    def get_action(self, state):
        # state = self.get_state(real_state)
        ru = np.random.uniform(0, 1)

        if ru < self.epsilon:
            action = np.random.randint(4)
        else:
            action = self.get_action_from_Q(state)

        return action

    def get_action_e(self, state, epsd):
        # state = self.get_state(real_state)
        ru = np.random.uniform(0, 1)

        if ru < epsd:
            action = np.random.randint(4)
        else:
            action = self.get_action_from_Q(state)

        return action

    def get_action_from_Q(self, state):
        action = 0
        options = [-1000,-1000,-1000,-1000] # np.zeros(4)
        for a in range(4):
            options[a] = self.get_q(state, a)
        action = np.argmax(options)
        return action

    # zwingend neu schreiben
    #
    def calc_policy(self):

        self.adjust_epsilon()

        self.states = []
        self.policy = {}
        for (s, a) in self.Q.keys():
            if s not in self.states:
                self.states.append(s)
        self.states.sort()
        for state in self.states:
            a0 = -100000
            a1 = -100000
            a2 = -100000
            a3 = -100000
            if (state, 0) in self.Q.keys():
                a0 = self.Q[(state,0)]
            if (state, 1) in self.Q.keys():
                a1 = self.Q[(state,1)]
            if (state, 2) in self.Q.keys():
                a2 = self.Q[(state,2)]
            if (state, 3) in self.Q.keys():
                a3 = self.Q[(state,3)]

            v = [a0, a1, a2, a3]

            a = np.argmax(v)
            self.policy[state] = a

    def adjust_q(self, state, action, reward, new_state, new_action):

        qsa = self.get_q(state, action)
        qsa_n = self.get_q(new_state, new_action)
        new_q = qsa + self.alpha * (reward + self.gamma * qsa_n - qsa)
        self.Q[(state, action)] = new_q
