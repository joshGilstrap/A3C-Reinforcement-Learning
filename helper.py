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
        self.prev_x_pos = 0
        self.prev_score = 0
        self.prev_time = 0
        self.stuck_iter = 0

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.env.seed(seed)
        self.prev_x_pos = 0
        self.prev_score = 0
        self.prev_time = 0
        self.stuck_iter = 0
        state = self.env.reset(**kwargs)
        return state, {}

    def step(self, actions):
        state, reward, done, info = self.env.step(actions)
        
        # right incentive
        distance = info['hpos'] - self.prev_x_pos
        reward += distance
        # if we're in the same place as last time, might be stuck
        if info['hpos'] == self.prev_x_pos:
            self.stuck_iter += 1
        self.prev_x_pos = info['hpos']
        
        # score incentive
        score_diff = info['score'] - self.prev_score
        reward += score_diff * 0.01
        self.prev_score = info['score']
        
        # time incentive
        time_diff = self.prev_time - info['time']
        reward -= time_diff
        self.prev_time = info['time']
        
        # reset on death or being stuck
        if info['lives'] < 4 or self.stuck_iter > 1000:
            reward -= 100
            done = True
        return state, reward, done, info

    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()