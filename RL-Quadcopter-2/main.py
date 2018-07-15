## TODO: Train your agent here.
import sys
import pandas as pd
from agents.policy_search import PolicySearch_Agent
from agents.agent import DDPG
from task import Task
from openai_task import OpenAITask
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tensorflow import set_random_seed

import gym

# seed

np.random.seed(0)
set_random_seed(0)


num_episodes = 200
target_pos = np.array([10, 10, 10.])
# task = Task(target_pos=target_pos)
env = gym.make('Pendulum-v0')
env.seed(0)
task = OpenAITask(env)
agent = DDPG(task)
print_every=5
scores_deque = deque(maxlen=print_every)
scores = []
best_score = -np.inf
for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    score = 0.0
    while True:
        task.env.render()
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        # agent.step(reward, done)
        agent.step(action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            scores_deque.append(score)
            scores.append(score)
            best_score = max(score, best_score)
            if i_episode % print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            break

    sys.stdout.flush()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
