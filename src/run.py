from environment import MeleeEnv
from agents.agent import Agent, Modes, Solvers
import matplotlib.pyplot as plt
import pandas as pd

def get_args(): # TODO: argparser
    solver = Solvers.DQN
    mode = Modes.UPDATE
    versus = 9 # CPU lvl where 0 is human
    plot_rewards = False
    return solver, mode, versus, plot_rewards

def run_test(agent):
    while True:
        agent.solver.run()

def run_train(agent, episodes, plot_rewards):
    rewards = load_rewards(agent.mode)
    for i in range(episodes): # hardcoded episodes
        reward = agent.solver.train_episode(i)
        if not reward:
            continue
        print('\nEpisode', i, 'reward:', reward, '\n')
        rewards.append(reward)
        save_rewards(rewards)
        if plot_rewards:
            plot(rewards, i+1==episodes)
        agent.solver.save()

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

def plot(rewards, final=False, smooth_window=20):
    fig = plt.figure(1)
    rewards = pd.Series(rewards)
    rewards_smoothed = rewards.rolling(
        smooth_window, min_periods=smooth_window
    ).mean()
    if final:
        plt.title("Result")
    else:
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return")
    plt.clf()
    plt.plot(rewards, label="Raw", c="b", alpha=0.3)
    if len(rewards_smoothed) >= smooth_window:
        plt.plot(
            rewards_smoothed,
            label=f"Smooth (win={smooth_window})",
            c="k",
            alpha=0.7,
        )
    plt.legend()
    if final:
        plt.show(block=True)
    else:
        plt.pause(0.001)

def main():
    solver, mode, versus, plot_rewards = get_args()
    env = MeleeEnv(opponent=versus)
    agent = Agent(env, solver, mode)
    if mode == Modes.TEST:
        run_test(agent)
    else:
        run_train(agent, 10000, plot_rewards) # 10,000 episodes

if __name__ == '__main__':
    main()