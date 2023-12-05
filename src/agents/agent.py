import argparser as ap
from agents.dqn import DQN
from agents.a3c import A3C
from argparser import Modes, Solvers

class Agent:
    def __init__(self, env, options):
        self.env = env
        self.mode = ap.getMode(options.mode)
        self.solverType = ap.getSolverType(options.solver)

        # Select agent solver
        if self.solverType == Solvers.DQN:
            # Uses options: alpha, gamma, epsilon, steps_per_episode, replay_memory_size, update_frequency, batch_size, layers
            self.solver = DQN(self.env, options)
        elif self.solverType == Solvers.A3C:
            # Uses options: alpha, gamma, epsilon
            self.solver = A3C(options)
        
        # Load trained model
        if self.mode != Modes.TRAIN and self.solverType != Solvers.A3C:
            self.solver.load()