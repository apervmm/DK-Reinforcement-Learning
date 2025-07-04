from dk_agent import DonkeyKongAgent


if __name__ == "__main__":
    # Initialize and run the agent with the specified model
    agent = DonkeyKongAgent(
        num_envs=1, render_mode="human", model_path="dk_agent_ppo_1m_wrapped"
    )
    agent.load()
    agent.play()
