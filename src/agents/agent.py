from enum import Enum
from agents.dqn import DQN, Hyperparameters

class Solvers(Enum):
    DQN = 1
    SAC = 2

class Modes(Enum):
    TRAIN = 1
    TEST = 2
    UPDATE = 3

class Agent:
    def __init__(self, env, solver, mode):
        self.env = env
        self.mode = mode

        # Select agent solver
        if solver == Solvers.DQN:
            self.options = Hyperparameters(alpha=0.01, gamma=0.95, epsilon=0.01, steps_per_episode=10000, replay_memory_size=2000, update_frequency=100, batch_size=32, layers=[32,32])
            self.solver = DQN(self.env, self.options)
        elif solver == Solvers.SAC:
            self.options = Hyperparameters(alpha=0.01, gamma=0.95, epsilon=0.01, steps_per_episode=10000, replay_memory_size=2000, update_frequency=100, batch_size=32, layers=[32,32])
            self.solver = DQN(self.env, self.options)
        
        # Load trained model
        if self.mode != Modes.TRAIN:
            self.solver.load()