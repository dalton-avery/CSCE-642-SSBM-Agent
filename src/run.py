from agents.dqn import DQN, Hyperparameters

from TheGym import MeleeEnv

options = Hyperparameters(alpha=0.01, gamma=0.95, epsilon=0.01, steps_per_episode=10000, replay_memory_size=2000, update_frequency=100, batch_size=32, layers=[32,32])
env = MeleeEnv()
agent = DQN(env, options)

for i in range(10000):
    reward = agent.train_episode(i)
    print('\nEpisode', i, 'reward:', reward, '\n')
    agent.save()