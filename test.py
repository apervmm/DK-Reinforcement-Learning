import os
import gymnasium as gym
import ale_py
import numpy as np
import pygame
import time

# Set up ROM path
rom_path = os.path.join(ale_py.__path__[0], "roms")
if not os.path.exists(rom_path):
    raise FileNotFoundError(f"ROMs not found at: {rom_path}\nDid you run AutoROM --accept-license?")

# Create the environment
env = gym.make("ALE/DonkeyKong-v5", obs_type="ram", render_mode="human")
obs, info = env.reset()
obs = np.array(obs)

pygame.init()
screen = pygame.display.set_mode((300, 100))
pygame.display.set_caption("Use Arrow Keys or Z to Control Mario")

key_to_action = {
    pygame.K_LEFT: 4,
    pygame.K_RIGHT: 3,
    pygame.K_UP: 2,
    pygame.K_z: 1,
}

print("Controls: ← → ↑ Z")

running = True
while running:
    action = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    for key, mapped_action in key_to_action.items():
        if keys[key]:
            action = mapped_action
            break

    obs, reward, terminated, truncated, info = env.step(action)
    obs = np.array(obs)

    print(f"Y candidates → RAM[5]: {obs[5]}, RAM[11]: {obs[11]}")

    time.sleep(0.03)

    if terminated or truncated:
        obs, info = env.reset()
        obs = np.array(obs)

pygame.quit()
env.close()
