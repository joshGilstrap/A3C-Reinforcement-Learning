# 

import retro
import torch
import cv2
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import StepAPICompatibility
from info import HyperParameters, Params

# Generalized Advantage Estimation -> https://arxiv.org/abs/1506.02438
def calculate_losses(rewards, action_probs, value_preds, entropy, masks, device="cuda", is_tuning=False):
    reward_len = len(rewards)
    advantages = torch.zeros(reward_len, HyperParameters["N_ENVS"], device=device)
    
    if not is_tuning:
        gamma = HyperParameters["GAMMA"]
        lmbda = HyperParameters["LMBDA"]
        ent_coef = HyperParameters["ENT_COEF"]
    else:
        gamma = Params["gamma"]
        lmbda = Params["lmbda"]
        ent_coef = Params["ent_coef"]
    
    gae = 0.0
    for r in reversed(range(reward_len - 1)):
        error = (rewards[r] + gamma * masks[r] * value_preds[r + 1] - value_preds[r])
        gae = error + gamma * lmbda * masks[r] * gae
        advantages[r] = gae
        critic_loss = advantages.pow(2).mean()
        actor_loss = (-(advantages.detach() * action_probs).mean() - ent_coef * entropy.mean())
    
    return critic_loss, actor_loss


def make_env(is_test):
    def _init():
        if is_test:
            env = retro.make(game=HyperParameters["GAME"])
        else:
            env = retro.make(game=HyperParameters["GAME"], render_mode='rgb_array')
        env = GameWrapper(env)
        env = RetroWrapper(env)
        env = StepAPICompatibility(env)
        return env
    return _init


class GameWrapper(gym.Wrapper):
    def __init__(self, env):
        super(GameWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
    
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        obs = self._preprocess(obs)
        return obs
    
    def step(self, actions):
        obs, reward, done, _, info = self.env.step(actions)
        obs = self._preprocess(obs)
        return obs, reward, done, info
    
    def _preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, dsize=(84, 84), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]
        return frame
    

class RetroWrapper(gym.Env):
    def __init__(self, env):
        super(RetroWrapper, self,).__init__()
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        self.prev_lives = 0
        self.prev_x_pos = 0
        self.prev_coins = 0
        self.stuck_count = 0

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.env.seed(seed)
        state = self.env.reset(**kwargs)
        return state, {}

    def step(self, actions):
        state, reward, done, info = self.env.step(actions)
        
        # Get ROM addresses
        memory = self.env.unwrapped.get_ram()
        # Mario's x position
        xpos = memory[0x086]
        info['xpos'] = xpos
        
        # Penalize life loss
        if info['lives'] < self.prev_lives:
            reward -= 100
        self.prev_lives = info['lives']
        
        # Reward ground covered
        if info['xpos'] > self.prev_x_pos:
            reward += info['xpos']
            self.stuck_count = 0
        # Penalize staying in place
        if info['xpos'] == self.prev_x_pos:
            reward -= 1
            self.stuck_count += 1
        self.prev_x_pos = info['xpos']
        
        # Reward haveing a score
        reward += info['score'] * 0.1
        # Reward getting coins
        if info ['coins'] > self.prev_coins:
            reward += info['coins']
        self.prev_coins = info['coins']
        
        # Stuck for good = reset
        if self.stuck_count > 500:
            done = True
        
        # Clip reward to prevent explosions
        reward = np.clip(reward, -10, 10)
        
        return state, reward, done, info

    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()