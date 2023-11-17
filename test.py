from src.TheGym import MeleeEnv
import time

env = MeleeEnv()
while True:
    time.sleep(.5)
    print(env.step(16))
    time.sleep(.5)
    print(env.step(17))