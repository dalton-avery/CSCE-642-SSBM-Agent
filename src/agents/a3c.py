class A3COptions:
    def __init__(self, alpha=0.01, gamma=0.95, epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

class A3C:
    def __init__(self, env, options):
        self.env = env
        self.options = options

    def train_episode(self):
        print("A3C Not implemented.")
        exit(0)