from agents.dqn import DQN, Hyperparameters

from TheGym import MeleeEnv
import matplotlib.pyplot as plt
import pandas as pd

options = Hyperparameters(alpha=0.01, gamma=0.95, epsilon=0.01, steps_per_episode=10000, replay_memory_size=2000, update_frequency=100, batch_size=32, layers=[32,32])
env = MeleeEnv()
episodes = 100000
agent = DQN(env, options)
rewards = []

def plot(episode, rewards, final=False, smooth_window=20):
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

for i in range(episodes):
    reward = agent.train_episode(i)
    if reward == None:
        continue
    print('\nEpisode', i, 'reward:', reward, '\n')
    rewards.append(reward)
    plot(i, rewards, i+1==episodes)
    agent.save()



