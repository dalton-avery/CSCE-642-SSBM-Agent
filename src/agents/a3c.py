import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from environment import MeleeEnv
from argparser import getMode, Modes


class GlobAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(GlobAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        # Shared layers
        for i in range(len(sizes) - 2):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        # Actor head layers
        self.layers.append(nn.Linear(hidden_sizes[-1], act_dim))
        # Critic head layers
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 2):
            x = F.relu(self.layers[i](x))

        # Actor head
        probs = F.softmax(self.layers[-2](x), dim=-1)
        # Critic head
        value = self.layers[-1](x)

        return torch.squeeze(probs, -1), torch.squeeze(value, -1)
    
    def loss_func(self, states, actions, target_values):
        self.train()
        probs, values = self.forward(states)
        td = target_values - values
        critic_loss = td.pow(2)

        m = torch.distributions.Categorical(probs)
        exp_v = m.log_prob(actions) * td.detach().squeeze()
        actor_loss = -exp_v
        total_loss = (critic_loss + actor_loss).mean()

        return total_loss
        

class A3CWorker(mp.Process):
    def __init__(self, options, id, global_episodes, global_adam, global_net):
        super(A3CWorker, self).__init__()
        self.id = id # used to determine port for slippi console
        self.options = options
        self.global_episodes = global_episodes
        self.optimizer = global_adam
        self.global_net = global_net

    def run(self):
        self.env = MeleeEnv(self.options.versus, port=51441+self.id)
        self.local_net = ActorCriticNetwork(self.env.get_obs_shape(), self.env.action_space.n, self.options.layers)
        reward_hist, prev_episodes = self.load_rewards(file_path=f'./src/agents/a3c/rewards-{self.id}.txt')
        with self.global_episodes.get_lock():
            self.global_episodes.value += prev_episodes
        while self.global_episodes.value < self.options.episodes:
            state, _ = self.env.reset()    
            buffer_s, buffer_a, buffer_r = [], [], []
            episode_reward = 0
            step_count = 1
            while self.options.steps_per_episode:    

                sampled_action = self.choose_action(state)
                if self.env.can_receive_action(): # only set new action if can receive new input
                    action = sampled_action
                    temp_action = action
                else:
                    temp_action = action
                    action = 24 # no op
                
                next_state, reward, done, _, _ = self.env.step(action)
                action = temp_action
                # print(f'Process {self.id} action: {action} reward: {reward}')
                
                episode_reward += reward
                buffer_a.append(action)
                buffer_s.append(state)
                buffer_r.append(reward)

                if step_count % self.options.update_frequency == 0 or done:
                    self.updateNetworks(self.local_net, self.global_net, next_state, done, buffer_s, buffer_a, buffer_r)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    if done:
                        with self.global_episodes.get_lock():
                            self.global_episodes.value += 1
                        break

                state=next_state

                if np.any((state < 0) | (state > 1)):
                    print(state)

                step_count += 1
            reward_hist.append(episode_reward)
            print('Episode:', self.global_episodes.value, 'Reward:', episode_reward, '\n')
            
            torch.save(self.global_net, "./src/agents/a3c/global.pt")
            self.save_rewards(reward_hist, f'./src/agents/a3c/rewards-{self.id}.txt')

    def choose_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        probs, value = self.local_net(state)
        probs_np = probs.detach().numpy()
        action = np.random.choice(len(probs_np), p=probs_np)
        return action
    
    def updateNetworks(self, local_net, global_net, next_state, done, buffer_s, buffer_a, buffer_r):
        # UPDATE LOCAL NETWORKS
        R = torch.as_tensor(0) if done else local_net.forward(torch.from_numpy(next_state))[-1]

        buffer_r_tar = []
        for r in buffer_r[::-1]:
            R = r + self.options.gamma * R
            buffer_r_tar.append(R.detach())
        buffer_r_tar.reverse()

        loss = local_net.loss_func(
        torch.from_numpy(np.vstack(buffer_s)),
        torch.from_numpy(np.vstack(buffer_a)),
        torch.from_numpy(np.array(buffer_r_tar)[:, None]))

        # UPDATE GLOBAL NETWORKS
        self.optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(local_net.parameters(), global_net.parameters()):
            gp._grad = lp.grad
        self.optimizer.step()

        # PULL GLOBAL INTO LOCAL
        local_net.load_state_dict(global_net.state_dict())

    def save_rewards(self, rewards, file_path='./src/agents/a3c/rewards.txt'):
        with open(file_path, 'w') as file:
            for reward in rewards:
                file.write(str(reward) + '\n')

    def load_rewards(self, file_path='./src/agents/a3c/rewards.txt'):
        rewards = []
        if getMode(self.options.mode) == Modes.UPDATE:
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    rewards = [float(line.strip()) for line in lines]
                    print("Loading previous rewards")
            except FileNotFoundError:
                print("File not found. Returning empty rewards list.")
        return rewards, len(rewards)

class A3C:
    def __init__(self, options):
        mp.set_start_method("spawn") # TODO: See if works without it on windows
        self.env = MeleeEnv(options.versus)
        self.options = options

        self.global_net = self.load_network()
        self.global_net.share_memory()
        
        self.global_adam = GlobAdam(self.global_net.parameters(), lr=self.options.alpha)

        self.global_episodes = mp.Value('i', 0)


    def start_workers(self):
        # Define Global params for Policy(actor) and Value(critic) networks
        self.workers = [A3CWorker(self.options, id, self.global_episodes, self.global_adam, self.global_net) for id in range(self.options.num_workers)]
        [worker.start() for worker in self.workers]
        [worker.join() for worker in self.workers]

    def load_network(self):
        if getMode(self.options.mode) == Modes.UPDATE:
            print('loading previous a3c model')
            return torch.load("./src/agents/a3c/global.pt")
        else:
            return ActorCriticNetwork(self.env.get_obs_shape(), self.env.action_space.n, self.options.layers)