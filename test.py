from src.TheGym import MeleeEnv

env = MeleeEnv()
while True:
    env.step(env.action_space.sample())