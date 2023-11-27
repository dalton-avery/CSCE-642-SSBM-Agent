from agents.dqn import DQN

from TheGym import MeleeEnv

env = MeleeEnv()
agent = DQN(env)

for i in range(100):
    agent.train_episode()