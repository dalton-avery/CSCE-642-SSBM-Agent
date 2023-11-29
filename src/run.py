from agents.dqn import DQN

from TheGym import MeleeEnv

env = MeleeEnv()
agent = DQN(env)

for i in range(10000):
    agent.train_episode(i)
    agent.save()