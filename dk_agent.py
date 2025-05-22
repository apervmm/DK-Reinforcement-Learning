import os
from typing import Callable
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium.envs.atari


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class DonkeyKongAgent:
    def __init__(self, render_mode=None, model_path="dk_agent_ram_mlp", num_envs=1):
        self.env_id = "ALE/DonkeyKong-v5"
        self.render_mode = render_mode
        self.model_path = model_path
        self.num_envs = num_envs
        self.device = self._detect_device()
        self.env = None
        self.model = None

    def _detect_device(self):
        if torch.cuda.is_available():
            print("Using CUDA")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using Apple MPS")
            return torch.device("mps")
        print("Falling back to CPU")
        return torch.device("cpu")

    def _make_ram_env(self, render_mode=None):
        return DummyVecEnv([lambda: gym.make(self.env_id, obs_type="ram", render_mode=render_mode)])

    def train(self, timesteps=100_000):
        self.env = self._make_ram_env(render_mode=None)
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            device=self.device,
            tensorboard_log="./dk_agent_logs/",
            learning_rate=linear_schedule(2.5e-4),
            clip_range=linear_schedule(0.1),
        )
        self.model.learn(total_timesteps=timesteps)
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}.zip")

    def load(self):
        self.env = self._make_ram_env(render_mode=self.render_mode)
        if os.path.exists(f"{self.model_path}.zip"):
            self.model = PPO.load(self.model_path, env=self.env, device=self.device)
            print("Model loaded successfully.")
        else:
            raise FileNotFoundError(f"No model found at {self.model_path}.zip")

    def evaluate(self, n_episodes=5):
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call train() or load() first.")
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=n_episodes)
        print(f"Mean reward: {mean_reward}, Std: {std_reward}")

    def play(self, n_episodes=1):
        vec_env = self._make_ram_env(render_mode="human")
        for _ in range(n_episodes):
            obs = vec_env.reset()
            done = [False]
            while not any(done):
                print(f"Y position: {obs[0][34]}")  # RAM byte 34 = Y pos
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
        vec_env.close()
