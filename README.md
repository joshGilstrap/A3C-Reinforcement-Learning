# Asynchronous Advanced Actor Critic Reinforcement Learning - Super Mario Bros 3

An implementation of the A3C reinforcement leanring algorithm trained to play Super Mario Bros 3.

Technologies used:
  - PyTorch
  - stable-retro
  - Gymnasium
  - Stable Baselines3
  - Optuna

## Features

- Customizable window size for Atari environments
- Integration with Pygame for rendering
- Easy-to-use interface for running and visualizing Atari games

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Installation Steps

1. **Clone the repository**:

    ```sh
    git clone https://github.com/yourusername/ataribot.git
    cd ataribot
    ```

2. **Create and activate a virtual environment (optional but recommended)**:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:

    ```sh
    pip install gymnasium pygame opencv-python
    ```

## Usage

### Running the AtariBot

Here's an example of how to use the AtariBot to run an Atari game with a custom window size:

```python
from ataribot import CustomAtariEnv

env_name = 'Pong-v4'
width, height = 800, 600  # Desired window size
custom_env = CustomAtariEnv(env_name, width, height)

obs = custom_env.reset()
done = False
while not done:
    action = custom_env.env.action_space.sample()
    obs, reward, done, info = custom_env.step(action)
    custom_env.render()
custom_env.close()
