from agents.dqn import DQN

from TheGym import MeleeEnv

env = MeleeEnv()
agent = DQN(env)

for i in range(10000):
    reward = agent.train_episode(i)
    print('\nEpisode', i, 'reward:', reward, '\n')
    agent.save()