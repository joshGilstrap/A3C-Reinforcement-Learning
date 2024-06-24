import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecTransposeImage
from a3c import Actor3Critic
from helper import make_env, calculate_losses
from info import HyperParameters
from params import get_params

def main():
    best = get_params()
    print(best)
    print()
    # Inline loop to create all environments
    envs = SubprocVecEnv([make_env(False) for _ in range(HyperParameters["N_ENVS"])])
    envs = VecFrameStack(envs, n_stack=4)
    envs = VecTransposeImage(envs)
    # Hard-coded state dimention. Needs to be one to match the single action input
    state_dim = 1
    # Size of a single action space. Describes all possible moves
    action_dim = envs.action_space.n
    # Computing device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # The agent the controls the critic and actor
    agent = Actor3Critic(state_dim, action_dim)
    agent.load_weights()
    writer = SummaryWriter(log_dir="./logs/")
    
    mapping = {}
    for index in range(2**action_dim):
        bin_action = np.array([int(x) for x in np.binary_repr(index, width=action_dim)])
        mapping[index] = bin_action
        
    # Main training loop. N_UPDATES = (Num Desired Timesteps)/(N_ENVS * N_STEPS)
    for phase in tqdm(range(HyperParameters["N_UPDATES"])):
        # Resetting tensors every phase to prevent gradient accumulation.
        phase_value_preds = torch.zeros(HyperParameters["N_STEPS"], HyperParameters["N_ENVS"], device=device)
        phase_rewards = torch.zeros(HyperParameters["N_STEPS"], HyperParameters["N_ENVS"], device=device)
        phase_log_probs = torch.zeros(HyperParameters["N_STEPS"], HyperParameters["N_ENVS"], device=device)
        masks = torch.zeros(HyperParameters["N_STEPS"], HyperParameters["N_ENVS"], device=device)
        
        # Only reset environments if we're on the first phase to intitialize
        if phase == 0:
            obs = envs.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        for step in range(HyperParameters["N_STEPS"]):
            # Grab action as well as logits and entropy
            actions, log_probs, entropy = agent.get_action(obs, False)
            actions = actions.cpu().numpy().tolist()
            actions = np.array([mapping[action] for action in actions])
            # Take a step through the environment, grabbing results
            _, rewards, dones, _ = envs.step(actions)
            # Run the current observation through the critic to get state values
            state_values = agent.get_value(obs, False)
                
            # Store the variables necessary to calculate loss
            phase_value_preds[step] = torch.squeeze(state_values)
            phase_rewards[step] = torch.tensor(rewards, device=device)
            phase_log_probs[step] = log_probs
            masks[step] = torch.tensor([not done for done in dones])
        
        # Grab the losses. Losses determine how well or poorly a model is performing.
        # The loss of both models in the agent are necessary to update them individually
        critic_loss, actor_loss = calculate_losses(phase_rewards, phase_log_probs, phase_value_preds, entropy, masks, device=device)
        # Update the neural netwroks critic and actor
        agent.update_nets(critic_loss, actor_loss)
        # Convert tensors to numpy arrays to append to the plotting lists
        critic_loss = critic_loss.detach().cpu().numpy()
        actor_loss = actor_loss.detach().cpu().numpy()
        
        # Tensorboard graphs creation
        writer.add_scalar('Actor Loss', actor_loss, phase)
        writer.add_scalar('Critic Loss', critic_loss, phase)
        running_rewards = 0
        for reward in phase_rewards:
            running_rewards += reward
            running_rewards = running_rewards / len(phase_rewards)
        writer.add_scalar('Average Rewards', running_rewards.mean(), phase)
    
    agent.save_model(agent)
    
if __name__ == "__main__":
    main()

