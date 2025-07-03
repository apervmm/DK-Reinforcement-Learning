# Donkey Kong Reinforcment Learning

## Preamble
This project is an implementation of a reinforcement learning agent for Donkey Kong, developed as part of CSC 480 at California Polytechnic State University. 

**Instructor**: Rodrigo Canaan

**Team Members**: Ryan Chan, Ryan Hu, Kelvin Villago, Sigourney Scott, Almas Perneshev



## Set-up
1. Install Python 3.10
2. `pip install -r requirements.txt`


## Running Demo
1. Update `model_path` in `play.py` to indicate the latest version of the saved model
2. To run the model, run `python3 play.py`


## Training the Model
1. Update the save file path in `train.py` to specify where the model should be saved
2. Train the model using `python3 train.py`


## Results Documentation
The 3 models tested, including DQN, PPO, and PPO with added hyperparameters, are specified in the `train()` function in the `DonkeyKongAgent` class. Executing `train.py` will train one of the three models and save them to the specified path. To view model performance, including the episode reward mean, install `tensorboard` using `pip install tensorboard`, and run the command `tensorboard --logdir <log-directory>` in the terminal. This will allow you to visualize the training progress and performance metrics of the agent. The log directory is also specified in `train.py`.


## Architecture
1. `dk_agent.py`: Contains the main logic for reinforcement learning, including environment setup and reward wrapper
2. `train.py`: Responsible for managing the save file, number of parallel environments, and timesteps during training.
3. `play.py`: Handles testing and validation of the trained model, including managing save files and environment parameters.



## References
1. Accessed May 2025. [Online]. Available: https://gymnasium.farama.org/v0.28.1/environments/atari/donkey kong/
2. P. Okzohen and J. Visser, “Learning to play donkey kong using neural networks and reinforcement learning,” 2017, accessed June 2025. [Online]. Available: https://fse.studenttheses.ub.rug.nl/15452/1/AI BA 2017 PAULOZKOHEN.pdf
3. D. Karunakaran, “The actor-critic reinforcement learning algorithm,” 2020, accessed June 2025. [Online]. Available: https://medium.com/intro-to-artificial-intelligence/the-actor-critic-reinforcement-learning-algorithm-c8095a655c14
4. K. Y. Chan, B. Abu-Salih, R. Qaddoura, A. M. Al-Zoubi, V. Palade, D.-S. Pham, J. D. Ser, and K. Muhammad, “Deep neural networks in the cloud: Review, applications, challenges and research directions,” Neurocomputing, vol. 545, p. 126327, 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0925231223004502
5. “ALE Documentation.” [Online]. Available: https://ale.farama.org/gymnasium-interface.html
6. “Tensor Attributes — PyTorch 2.7 documentation.” [Online]. Available: https://docs.pytorch.org/docs/stable/tensor attributes.html#torch.device
7. “Welcome to Stable Baselines docs! - RL Baselines Made Easy — Stable Baselines 2.10.3a0 documentation.” [Online]. Available: https://stable-baselines.readthedocs.io/en/master/index.html
8. “DLR-RM/rl-baselines3-zoo,” Jun. 2025, original-date: 2020-05-05T05:53:27Z. [Online]. Available: https://github.com/DLR-RM/rl-baselines3-zoo

# Developers
Ryna Hu, Ryan Chan, Almas Perneshev, Sigourney Scott, Celvin Villago
