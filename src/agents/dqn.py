import random
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch.optim import AdamW

class QFunction(nn.Module):
    """
    Q-network definition.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
    ):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(dim=-1)

class DQN:
    def __init__(self, env, options):
        self.env = env(options.versus)
        self.options = options
        # Create Q-network
        self.model = QFunction(
            self.env.get_obs_shape(),
            self.env.action_space.n,
            self.options.layers,
        )
        # Create target Q-network
        self.target_model = deepcopy(self.model)
        # Set up the optimizer
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.options.alpha, amsgrad=True
        )
        # Define the loss function
        self.loss_fn = nn.SmoothL1Loss()

        # Freeze target network parameters
        for p in self.target_model.parameters():
            p.requires_grad = False

        # Replay buffer
        self.replay_memory = deque(maxlen=self.options.replay_memory_size)

        # Number of episodes
        self.episodes = 0

        # Number of training steps so far
        self.n_steps = 0
    
    def save(self):
        torch.save(self.model, "./src/agents/dqn/model.pt")
        torch.save(self.target_model, "./src/agents/dqn/target_model.pt")

    def load(self):
        print('Loading previous models')
        self.model = torch.load("./src/agents/dqn/model.pt")
        self.target_model = torch.load("./src/agents/dqn/target_model.pt")

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        self.options.epsilon = math.e**(-0.001 * self.episodes)

    def epsilon_greedy(self, state):
        nA = self.env.action_space.n

        aStar = torch.argmax(self.model(torch.as_tensor(state)))

        probs = np.zeros((nA,), dtype=float)
        for a in range(nA):
            if a == aStar:
                probs[a] = 1 - self.options.epsilon + self.options.epsilon / nA
            else:
                probs[a] = self.options.epsilon / nA

        return probs

    def compute_target_values(self, next_states, rewards, dones):
        target_q = torch.zeros(len(dones)) 

        for i in range(len(dones)):
            if dones[i]:
                target_q[i] = rewards[i]
            else:
                target_q[i] = rewards[i] + self.options.gamma * torch.max(self.target_model(next_states[i]))

        return target_q

    def replay(self):
        if len(self.replay_memory) > self.options.batch_size:
            minibatch = random.sample(self.replay_memory, self.options.batch_size)
            minibatch = [
                np.array(
                    [
                        transition[idx]
                        for transition, idx in zip(minibatch, [i] * len(minibatch))
                    ]
                )
                for i in range(5)
            ]
            states, actions, rewards, next_states, dones = minibatch
            # Convert numpy arrays to torch tensors
            states = torch.as_tensor(states, dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.float32)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            next_states = torch.as_tensor(next_states, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)

            # Current Q-values
            current_q = self.model(states)
            # Q-values for actions in the replay memory
            current_q = torch.gather(
                current_q, dim=1, index=actions.unsqueeze(1).long()
            ).squeeze(-1)

            with torch.no_grad():
                target_q = self.compute_target_values(next_states, rewards, dones)

            # Calculate loss
            loss_q = self.loss_fn(current_q, target_q)

            # Optimize the Q-network
            self.optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()
    
    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))
    
    def train_episode(self):
        self.update_epsilon()
        state, _ = self.env.reset()
        action = 24 # start with no-op
        total_reward = 0
        for step in range(self.options.steps_per_episode):
            sampled_action = np.random.choice(np.arange(self.env.action_space.n), p=self.epsilon_greedy(state)) # try to input next action
            if self.env.can_receive_action(): # only set new action if can receive new input
                action = sampled_action
            next_state, reward, done, _, _ = self.env.step(action)
            total_reward += reward
            self.memorize(state, action, reward, next_state, done)
            state = next_state
            self.replay()
            if step % self.options.update_frequency == 0:
                self.update_target_model()
            if done:
                break
        self.episodes += 1
        return total_reward
    
    def create_greedy_policy(self):
        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            q_values = self.model(state)
            return torch.argmax(q_values).detach().numpy()

        return policy_fn
    
    def run(self):
        state = self.env._get_flat_state(self.env._get_obs())
        probs = self.epsilon_greedy(state)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        self.env.step(action)

    # CHANGES
    # Layers: 64, 64, 64
    # Reward function tweaking
    # Make action taken reflect action currently doing
