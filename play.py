from dk_agent import DonkeyKongAgent


if __name__ == "__main__":
    agent = DonkeyKongAgent(num_envs=4, render_mode="human")
    agent.load()
    agent.play()
