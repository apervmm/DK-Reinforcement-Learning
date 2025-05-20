from dk_agent import DonkeyKongAgent


if __name__ == "__main__":
    agent = DonkeyKongAgent(
        num_envs=1, render_mode="human", model_path="dk_agent_ppo_1m_zoo_tuning"
    )
    agent.load()
    agent.play()
