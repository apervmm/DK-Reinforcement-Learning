#Donkey Kong Reinforcment Learning

##Set-up:
1. Install Python 3.10
2. pip install -r requirements.txt


## Running Demo:
- ***play.py*** Indicate the latest version of the saved model in `model_path`
- ***Run:*** python3 play.py



##Training own model:
- ***train.py** incdicate the file where to save the model
- ***Run:** python3 train.py


##Architecture:
1. ***dk_agent.py:*** This files includes the main logic of Reinforcment learning, setting up the environment, and reward wrapper
2. ***train.py:*** Training script/executable of the program, which responsible for choosing managing the save file, number of parallel environments, and timesteps
3. ***play.py:*** Testing script/executable of the program, which responsible for choosing managing the save file, number of parallel environments, and timesteps


