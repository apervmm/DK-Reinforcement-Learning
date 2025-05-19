from dk_agent import DonkeyKongAgent


if __name__ == "__main__":
    agent = DonkeyKongAgent(num_envs=4)
    agent.train(timesteps=1_000_000)
    agent.evaluate()
