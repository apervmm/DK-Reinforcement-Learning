from dk_agent import DonkeyKongAgent


if __name__ == "__main__":
    agent = DonkeyKongAgent(num_envs=8)
    agent.train(timesteps=20_000)
    agent.evaluate()
