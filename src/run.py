from agents.dqn import DQN

from TheGym import MeleeEnv

env = MeleeEnv()
agent = DQN(env)

agent.train_episode()