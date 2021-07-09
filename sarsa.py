import numpy as np

class Sarsa:

    alpha = 0.2
    gamma = 0.7
    epsilon = 0.7
    eps_min = 0.1
    eps_f = 0.99

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
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, 4)
        else:
            if state in self.policy:
                action = self.policy[state]
            else:
                action = np.random.randint(0, 4)

        return action

    def get_action_from_policy(self, state):
        action = self.policy[state]
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

    def adjust_q(self, state, action, new_state, reward):

        new_action = self.get_action(new_state)

        self.Q[(state, action)] = self.get_q(state, action) + self.alpha * (
                reward + self.gamma * self.get_q(new_state, new_action) - self.get_q(state, action))
