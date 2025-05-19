import os
import ale_py
import torch
import gymnasium as gym
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy


class DonkeyKongAgent:
    def __init__(self, render_mode=None, model_path="dk_agent", num_envs=1):
        self.env_id = "ALE/DonkeyKong-v5"
        self.render_mode = render_mode
        self.model_path = model_path
        self.num_envs = num_envs
        self.device = self._detect_device()
        self.env = self._make_vector_envs()
        self.model = None
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

    def _make_vector_envs(self):
        env = make_atari_env(
            self.env_id,
            n_envs=self.num_envs,
            seed=0,
            env_kwargs={"render_mode": self.render_mode},
        )
        env = VecFrameStack(env, n_stack=4)
        return env

    def train(self, timesteps=100_000):
        # self.model = DQN(
        #     "CnnPolicy",
        #     self.env,
        #     verbose=0,
        #     buffer_size=100_000,
        #     learning_starts=50_000,
        #     exploration_fraction=0.1,
        #     exploration_final_eps=0.05,
        #     tensorboard_log="./dk_agent_logs/",
        #     device=self.device,
        # )
        self.model = DQN(
            "CnnPolicy",
            self.env,
            verbose=1,
            device=self.device,
            tensorboard_log="./dk_agent_logs/",
        )
        self.model.learn(total_timesteps=timesteps)
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

    def load(self):
        if os.path.exists(f"{self.model_path}.zip"):
            self.model = DQN.load(self.model_path, env=self.env)
            print("Model loaded successfully.")
        else:
            raise FileNotFoundError(f"No model found at {self.model_path}")

    def evaluate(self, n_episodes=5):
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call train() or load() first.")
        mean_reward, std_reward = evaluate_policy(
            self.model, self.env, n_eval_episodes=n_episodes
        )
        print(f"Mean reward: {mean_reward}, Std: {std_reward}")

    def play(self, n_episodes=1):
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call train() or load() first.")

        def make_env():
            env = gym.make(self.env_id, render_mode="human")
            return AtariWrapper(env)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecFrameStack(vec_env, n_stack=4)

        for _ in range(n_episodes):
            obs = vec_env.reset()
            done = [False]
            while not any(done):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
        vec_env.close()
