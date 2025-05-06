import gymnasium as gym
import ale_py
from gymnasium.wrappers import FlattenObservation

env = gym.make("ALE/DonkeyKong-v5", render_mode="human")

env.observation_space.shape
(96, 96, 3)
wrapped_env = FlattenObservation(env)
wrapped_env.observation_space.shape

observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()