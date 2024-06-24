import optuna
import torch
import retro
import numpy as np
from tqdm import tqdm
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecTransposeImage
from helper import RetroWrapper, GameWrapper
from gymnasium.wrappers import StepAPICompatibility
from info import HyperParameters, Params
from helper import make_env, calculate_losses
from a3c import Actor3Critic

def optimize_params(trial):
    # Mimic environment set up from training
    envs = SubprocVecEnv([make_env(False) for _ in range(HyperParameters["N_ENVS"])])
    envs = VecFrameStack(envs, n_stack=4)
    envs = VecTransposeImage(envs)
    
    # Specify ranges and variables for all desired parameters
    n_steps = trial.suggest_int('n_steps', 5, 20)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    critic_lr = trial.suggest_float('critic_lr', 1e-5, 2.5e-4, log=True)
    actor_lr = trial.suggest_float('actor_lr', 1e-6, 1e-4, log=True)
    ent_coef = trial.suggest_float('ent_coef', 0.01, 0.1)
    lmbda = trial.suggest_float('lmbda', 0.9, 1.0)
    epsilon = trial.suggest_float('epsilon', 1e-8, 1e-2, log=True)
    grad_norm = trial.suggest_float('grad_norm', 0.1, 40)
    
    # Populate dict with current values
    Params['n_steps'] = n_steps
    Params['gamma'] = gamma
    Params['critic_lr'] = critic_lr
    Params['actor_lr'] = actor_lr
    Params['ent_coef'] = ent_coef
    Params['lmbda'] = lmbda
    Params['epsilon'] = epsilon
    Params['grad_norm'] = grad_norm
    
    # Mimic agent set up from training
    state_dim = 1
    action_dim = envs.action_space.n
    device = "cpu"
    agent = Actor3Critic(state_dim, action_dim, is_tuning=True)
    agent.load_weights()
    
    # Mimic binary action space compatibility from training
    mapping = {}
    for index in range(2**action_dim):
        bin_action = np.array([int(x) for x in np.binary_repr(index, width=action_dim)])
        mapping[index] = bin_action
    
    # Mimic main training loop
    for phase in tqdm(range(313)):
        phase_value_preds = torch.zeros(Params['n_steps'], HyperParameters["N_ENVS"], device=device)
        phase_rewards = torch.zeros(Params['n_steps'], HyperParameters["N_ENVS"], device=device)
        phase_log_probs = torch.zeros(Params['n_steps'], HyperParameters["N_ENVS"], device=device)
        masks = torch.zeros(Params['n_steps'], HyperParameters["N_ENVS"], device=device)
        
        if phase == 0:
            obs = envs.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        for step in range(Params['n_steps']):
            actions, log_probs, entropy = agent.get_action(obs, False)
            actions = actions.cpu().numpy().tolist()
            actions = np.array([mapping[action] for action in actions])
            _, rewards, dones, _ = envs.step(actions)
            state_values = agent.get_value(obs, False)
            
            phase_value_preds[step] = torch.squeeze(state_values)
            phase_rewards[step] = torch.tensor(rewards, device=device)
            phase_log_probs[step] = log_probs
            masks[step] = torch.tensor([not done for done in dones])
            
        critic_loss, actor_loss = calculate_losses(phase_rewards, phase_log_probs, phase_value_preds, entropy, masks, device=device, is_tuning=True)
        agent.update_nets(critic_loss, actor_loss)
    
    envs.close()
    
    # Mimic testing of trained model
    env = retro.make(game=HyperParameters["GAME"], render_mode='rgb_array')
    env = GameWrapper(env)
    env = RetroWrapper(env)
    env = StepAPICompatibility(env)
    
    for _ in range(10):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0)
        
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(state, True)
                action = action.cpu().numpy().tolist()
                action = np.array([mapping[act] for act in action])
                action = action.flatten()
            
            state, reward, term, _, _ = env.step(action)
            total_reward += reward
            done = term
    env.close()
    
    return (total_reward / 10)

def get_params():
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_params, n_trials=50)
    optimal = study.best_params
    return optimal