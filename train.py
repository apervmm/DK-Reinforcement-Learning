from dk_agent import DonkeyKongAgent


if __name__ == "__main__":
    agent = DonkeyKongAgent(num_envs=8, model_path="dk_agent_ppo_1m_tuned")
    agent.train(timesteps=1_000_000)
    agent.evaluate()
