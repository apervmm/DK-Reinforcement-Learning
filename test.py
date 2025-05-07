import gymnasium as gym
import ale_py

env = gym.make("ALE/DonkeyKong-v5", render_mode="human")

# Uncomment the following lines to see the shape of the observation space before and after flattening
# print(env.observation_space.shape)
# wrapped_env = FlattenObservation(env)
# print(wrapped_env.observation_space.shape)

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
