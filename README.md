# Asynchronous Advanced Actor Critic Reinforcement Learning - Super Mario Bros 3

An implementation of the A3C reinforcement leanring algorithm trained to play Super Mario Bros 3.

Technologies used:
  - PyTorch
  - stable-retro
  - Gymnasium
  - Stable Baselines3
  - Optuna

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

### Installation Steps

1. **Clone the repository**:

    ```sh
    git clone https://github.com/<your username>/Asynchronous-Advanced-Actor-Critic.git
    ```

2. **Create and activate a virtual environment (optional but recommended)**:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

**Training Your Own Model**
  1. Optimize Hyperparamters (optional)
     - Uncomment lines 12-14
     - Place a breakpoint at line 14
     - Run the debugger
     - Copy down best trials values into Hyperparameters in info.py
     - If you don't want to optimize the parameters simply uncomment lines 1-15 in info.py
         and comment out lines 17-31
  2. Comment out lines 12-14 of main.py
  3. Comment out line 28 main.py if not already to avoid loading pretrained model weights
  4. Run the program
     - The software automatically saves the models weights and optimizers after training is complete

**Load A Pretrained Model**
  1. Uncomment line 28 if not already
  2. Make sure the filepaths on lines 111-114 of a3c.py match the directories of the intended model
     in /checkpoints
  3. Run test.py
    
**Train On A Different Game**
  1. Make sure to load your legally obtained ROM into a stable-retro environment
     - Open a command prompt at the directory you ROM is in
     - Use command ```python3 -m retro.import /path/to/your/ROMs/directory/```
     - You will see a message with how many games were imported, only compatible ROMS will import.
         Look [here](https://stable-retro.farama.org/getting_started/) for more info
  2. Replace the name of the game in the Hyperparameters dictionary in info.py
  3. Make a custom reward function based off of info
     - The reward functionality in the step function of RetroWrapper in helper.py is SMB3 specific.
     - In order to train effectively (or at all) you need to give game-specific circumstances custom weights.
     - Here's a few steps to see if there's any built-in information to use:
       1. Place a breakpoint anywhere after a step() function call, line 61 of helper.py is a good one
       2. Run the debugger
       3. Check what's returned from step() in "info". These are variables updated everytime step is called
         - This info can be used to create done conditions, game goal incentives, lives count, and more
         - For example, SMB3 comes with 4 built-in variables: score, lives, hpos (horizontal position), time
       4. Give numerical values to represent positives/penaltys and add them to the reward variable

## License

[MIT License](https://github.com/joshGilstrap/Asynchronous-Advanced-Actor-Critic/blob/main/LICENSE)
