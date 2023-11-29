from agents.dqn import DQN

from TheGym import MeleeEnv

env = MeleeEnv() # initiialize environment
agent = DQN(env) # initialize agent

for i in range(100):
    agent.train_episode(i)