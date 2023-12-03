import argparse
from enum import Enum

class Solvers(Enum):
    DQN = 1
    A3C = 2

class Modes(Enum):
    TRAIN = 1
    TEST = 2
    UPDATE = 3


def parseArgs():
    parser = argparse.ArgumentParser(description='Example of libmelee in action')

    parser.add_argument('--alpha', '-a', default=0.01, type=float, help='Alpha value (learning rate)')
    parser.add_argument('--gamma', '-g', default=0.95, type=float, help='Gamme value (discount factor)')
    parser.add_argument('--epsilon', '-e', default=0.01, type=float, help='Epsilon (exploration rate)')
    parser.add_argument('--episodes', '-ep', default=5000, type=int, help='Episodes to train')
    parser.add_argument('--steps_per_episode', '-st', default=10000, type=int, help='Steps per episode')
    parser.add_argument('--replay_memory_size', '-rs', default=2000, type=int, help='Replay Memory Size')
    parser.add_argument('--update_frequency', '-uf', default=10001, type=int, help='Fequency to update model (exclude for per episode)')
    parser.add_argument('--batch_size', '-b', default=32, type=int, help='Batch Size for learning')
    parser.add_argument('--layers', '-l', default=[64,64,64], help='Layer size for model')
    parser.add_argument('--solver', '-s', default='a3c', help='Choose learning agent (dqn or a3c)')
    parser.add_argument('--mode', '-m', default='train', help='Choose training mode (train, update, test)')
    parser.add_argument('--versus', '-v', default=9, help='The level of cpu to train against (0 for human)')
    parser.add_argument('--num_workers', '-n', default=4, type=int, help='The number of workers for multiagent solvers (A3C)')
    

    args = parser.parse_args()

    return args

def getSolverType(solverString):
    if solverString == 'dqn':
        return Solvers.DQN
    else:
        return Solvers.A3C
    
def getMode(modeString):
    if modeString == 'train':
        return Modes.TRAIN
    elif modeString == 'update':
        return Modes.UPDATE
    else:
        return Modes.TEST