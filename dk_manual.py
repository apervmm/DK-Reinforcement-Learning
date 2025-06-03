import gymnasium as gym
import ale_py
import pygame
import time
from stable_baselines3.common.atari_wrappers import AtariWrapper


def main():
    pygame.init()
    screen = pygame.display.set_mode((200, 200))  # Dummy window for key events
    pygame.display.set_caption("Donkey Kong RAM Diff Viewer")

    env = gym.make("ALE/DonkeyKong-v5", render_mode="human", frameskip=1)
    env = AtariWrapper(env)

    obs, info = env.reset()
    done = False

    print("Use ← ↑ ↓ → and SPACE to play. Press ESC to quit.")

    key_action_map = {
        pygame.K_LEFT: 4,  # LEFT
        pygame.K_RIGHT: 3,  # RIGHT
        pygame.K_UP: 2,  # UP
        pygame.K_DOWN: 5,  # DOWN
        pygame.K_SPACE: 1,  # FIRE
    }

    clock = pygame.time.Clock()
    step_num = 0
    prev_ram = env.unwrapped.ale.getRAM()

    while not done:
        action = 0  # NOOP

        # Poll events to handle ESC / quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                done = True

        # Continuous key state check
        keys = pygame.key.get_pressed()
        for key, mapped_action in key_action_map.items():
            if keys[key]:
                action = mapped_action
                break  # take first held key

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        curr_ram = env.unwrapped.ale.getRAM()
        changed = [
            (i, prev_ram[i], curr_ram[i])
            for i in range(128)
            if prev_ram[i] != curr_ram[i]
        ]

        if changed:
            print(f"\nStep {step_num:03d} — {len(changed)} RAM values changed:")
            for i, before, after in changed:
                print(f"  RAM[{i:>3}] = {before:>3} → {after:>3}")

        prev_ram = curr_ram.copy()
        step_num += 1
        clock.tick(15)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
