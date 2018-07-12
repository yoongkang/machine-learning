import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.5
        self.episodes = 0
        self.gamma = 0.9

    def select_action(self, state, increment=True):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if increment:
            self.episodes += 1
        probs = self.policy(state)
        return np.random.choice(self.nA, p=probs)

    def policy(self, state):
        epsilon = 1.0 / self.episodes
        probs = np.ones(self.nA) * epsilon / self.nA
        max_a = np.argmax(self.Q[state])
        probs[max_a] = 1 - epsilon + epsilon / self.nA
        return probs

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        next_action = self.select_action(next_state, increment=False)
        self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
