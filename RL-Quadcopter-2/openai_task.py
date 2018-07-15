import numpy as np
from physics_sim import PhysicsSim

class OpenAITask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, env):
        # Simulation
        self.env = env

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_high = env.action_space.high
        self.action_low = env.action_space.low

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.action_repeat = 1


    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    def reset(self):
        """Reset the sim to start a new episode."""
        state = self.env.reset()
        return state

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
