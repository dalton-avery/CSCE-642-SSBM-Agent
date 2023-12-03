from environment import MeleeEnv
from agents.agent import Agent
from argparser import Modes, Solvers
import argparser as ap


def run_test(agent):
    while True:
        agent.solver.run()

def run_train(agent, episodes):
    rewards = load_rewards(agent.mode)
    if agent.solverType == Solvers.DQN:
        for i in range(episodes): # hardcoded episodes
            reward = agent.solver.train_episode()
            if not reward:
                continue
            print('\nEpisode', i, 'reward:', reward, '\n')
            rewards.append(reward)
            save_rewards(rewards)
            agent.solver.save()
    else: 
        agent.solver.start_workers()


def save_rewards(rewards, file_path='./src/agents/dqn/rewards.txt'):
    with open(file_path, 'w') as file:
        for reward in rewards:
            file.write(str(reward) + '\n')

def load_rewards(agent_mode, file_path='./src/agents/dqn/rewards.txt'):
    rewards = []
    if agent_mode == Modes.UPDATE:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                rewards = [float(line.strip()) for line in lines]
                print("Loading previous rewards")
        except FileNotFoundError:
            print("File not found. Returning empty rewards list.")
    
    return rewards

def main():
    options = ap.parseArgs()
    env = MeleeEnv
    agent = Agent(env, options)
    if options.mode == Modes.TEST:
        run_test(agent)
    else:
        run_train(agent, options.episodes) # 10,000 episodes

if __name__ == '__main__':
    main()