from dk_agent import DonkeyKongAgent


if __name__ == "__main__":
    agent = DonkeyKongAgent(num_envs=8, model_path="dk_agent_ram_mlp")
    agent.train(timesteps=20_000)
    agent.evaluate()
