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
The 3 models tested, including DQN, PPO, and PPO with added hyperparameters, are specified in the `train()` function in the `DonkeyKongAgent` class. Executing `train.py` will train one of the three models and save them to the specified path.


## Architecture
1. `dk_agent.py`: Contains the main logic for reinforcement learning, including environment setup and reward wrapper
2. `train.py`: Responsible for managing the save file, number of parallel environments, and timesteps during training.
3. `play.py`: Handles testing and validation of the trained model, including managing save files and environment parameters.



