
class DQN():
    def __init__(self, env):
        self.env = env

    def update_target_model(self):
        pass

    def epsilon_greedy(self):
        pass

    def compute_target_values(self):
        pass

    def replay(self):
        pass
    
    def memorize(self):
        pass
    
    def train_episode(self):
        while True:
            obs, r, _, _, _ = self.env.step(self.env.action_space.sample())
    
    def plot(self):
        pass
    
    def create_greedy_policy(self):
        pass
