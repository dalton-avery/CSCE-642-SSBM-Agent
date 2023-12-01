from environment import MeleeEnv
from agents.agent import Agent, Modes, Solvers

def get_args(): # TODO: argparser
    solver = Solvers.DQN
    mode = Modes.TRAIN
    versus = 9 # CPU lvl where 0 is human
    return solver, mode, versus

def run_test(agent):
    while True:
        agent.solver.run()

def run_train(agent):
    for i in range(10000): # hardcoded episodes
        reward = agent.solver.train_episode(i)
        print('\nEpisode', i, 'reward:', reward, '\n')
        agent.solver.save()

def main():
    solver, mode, versus = get_args()
    env = MeleeEnv(opponent=versus)
    agent = Agent(env, solver, mode)
    if mode == Modes.TEST:
        run_test(agent)
    else:
        run_train(agent)

if __name__ == '__main__':
    main()