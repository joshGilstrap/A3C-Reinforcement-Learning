import torch
import retro
import numpy as np
from gymnasium.wrappers import StepAPICompatibility
from helper import GameWrapper, RetroWrapper
from info import HyperParameters
from a3c import Actor3Critic

env = retro.make(game=HyperParameters["GAME"])
env = GameWrapper(env)
env = RetroWrapper(env)
env = StepAPICompatibility(env)

agent = Actor3Critic(1, 9)
agent.load_weights(False)

action_dim = 9
mapping = {}
for index in range(2**action_dim):
    bin_action = np.array([int(x) for x in np.binary_repr(index, width=action_dim)])
    mapping[index] = bin_action

for cycle in range(100):
    print(f"Cycle {cycle + 1} starting...")
    
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0)
    
    done = False
    while not done:
        with torch.no_grad():
            action, _, _ = agent.get_action(state, True)
            action = action.cpu().numpy().tolist()
            action = np.array([mapping[act] for act in action])
            action = action.flatten()
        
        state, reward, term, _, info = env.step(action)
        done = term