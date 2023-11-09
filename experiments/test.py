# Main script for simulations.
# Author: @THEFFTKID.

# Gym stuff
import gymnasium as gym
import environment as ENV

# Stable baselines
from stable_baselines3 import A2C

# Other libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data.
# TODO: Add data reader call.
df = pd.read_csv('experiments/SPY_data.csv')

# Select just a small subset of data.
df = df.iloc[:10]

# 2. Creation of the environment.
env = ENV.TradingEnv(df=df, window_size=5)

print("observation_space:", env.observation_space)

import pdb
pdb.set_trace()

# 4. Train Environment
model = A2C('MlpPolicy', env, verbose=0) 
model.learn(total_timesteps=50, progress_bar=True)

# 5. Test Environment
observation, info = env.reset()

actions = []

import pdb
pdb.set_trace()


while True:
    # Get the policy action from an observation.
    action, _states = model.predict(observation)

    actions.append(actions)

    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
        break

env.close()
print("info:", info)
# plt.figure(figsize=(16, 6))
# env.unwrapped.render_all()
# plt.show()