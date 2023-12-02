import matplotlib.pyplot as plt
import pandas as pd
from agents.agent import Modes

def load_rewards(file_path='./src/agents/dqn/rewards.txt'):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            rewards = [float(line.strip()) for line in lines]
            print("Loading rewards")
            plot(rewards)
    except FileNotFoundError:
        print("File not found. Returning empty rewards list.")
    
    return rewards

def plot(rewards, smooth_window=20):
    fig = plt.figure(1)
    rewards = pd.Series(rewards)
    rewards_smoothed = rewards.rolling(
        smooth_window, min_periods=smooth_window
    ).mean()

    plt.title("Result")

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
    plt.show(block=True)

if __name__ == '__main__':
    load_rewards()