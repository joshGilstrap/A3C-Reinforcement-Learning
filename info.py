# HyperParameters = {
#     "GAME" : "SuperMarioBros3-Nes",
#     "CRITIC_LR" : 2.5e-4,
#     "ACTOR_LR" : 1e-4,
#     "N_ENVS" : 16,
#     # 10,000,000 (desired timesteps) / (N-ENVS * N_STEPS) = N_UPDATES
#     "N_UPDATES" : 31250,
#     "N_STEPS" : 20,
#     "GAMMA" : 0.999,
#     "LMBDA" : 0.9,
#     "EPSILON" : 1e-5, # 1e-8 defaut
#     "ENT_COEF" : 0.08,
#     "SAVE_EVERY" : 1000,
#     "GRAD_NORM" : 8.0,
# }

# SMB3
# HyperParameters = {
#     "GAME" : "SuperMarioBros3-Nes",
#     "CRITIC_LR" : 0.00023526461299240228,
#     "ACTOR_LR" : 1.142725757244386e-06,
#     "N_ENVS" : 16,
#     # 10,000,000 (desired timesteps) / (N-ENVS * N_STEPS) = N_UPDATES
#     "N_UPDATES" : 62500,
#     "N_STEPS" : 10,
#     "GAMMA" : 0.9311158224467578,
#     "LMBDA" : 0.9984790124513077,
#     "EPSILON" : 0.0004790167604033657, # 1e-8 defaut
#     "ENT_COEF" : 0.06690435135178872,
#     "SAVE_EVERY" : 1000,
#     "GRAD_NORM" : 25.870555304462748,
# }
# SMB
HyperParameters = {
    "GAME" : "SuperMarioBros-Nes",
    "CRITIC_LR" : 3.102183218722325e-05,
    "ACTOR_LR" : 2.982815080547892e-06,
    "N_ENVS" : 16,
    # 10,000,000 (desired timesteps) / (N-ENVS * N_STEPS) = N_UPDATES
    "N_UPDATES" : 4664,
    "N_STEPS" : 134,
    "GAMMA" : 0.9484013660083809,
    "LMBDA" : 0.9542934050057869,
    "EPSILON" : 8.527877899931554e-05, # 1e-8 defaut
    "ENT_COEF" : 0.018630272880004288,
    "SAVE_EVERY" : 1000,
    "GRAD_NORM" : 0.99195413382146,
}

Params = {}


# SMB3 reward function
        # memory = self.env.unwrapped.get_ram()
        # ypos = memory[0x0228]
        # info['ypos'] = ypos
        
        # if self.start_x_pos == 0:
        #     self.start_x_pos = info['hpos']
        
        # if self.start_y_pos == 0:
        #     self.start_y_pos = info['ypos']
        
        # # right incentive
        # distance_from_start = info['hpos'] - self.start_x_pos
        # distance = info['hpos'] - self.prev_x_pos
        # reward += distance_from_start * 0.1
        # reward += distance * 0.1
        # # if we're in the same place as last time, might be stuck
        # if info['hpos'] == self.prev_x_pos:
        #     self.stuck_iter += 1
        # else:
        #     self.stuck_iter = 0
        # self.prev_x_pos = info['hpos']
        
        # # score incentive
        # score_diff = info['score'] - self.prev_score
        # reward += score_diff * 0.01
        # self.prev_score = info['score']
        
        # # time incentive
        # time_diff = self.prev_time - info['time']
        # reward += time_diff * 0.01
        # self.prev_time = info['time']
        
        # # height_diff = abs(self.prev_y_pos - info['ypos'])
        # start = self.start_y_pos
        # height_diff = int(start) - info['ypos']
        # reward += height_diff * 0.1
        
        # # reset on death or being stuck
        # if info['lives'] < 4 or self.stuck_iter > 500:
        #     reward -= 100
        #     done = True

# SMB3 __init__ variables
        # self.prev_x_pos = 0
        # self.prev_y_pos = 0
        # self.start_x_pos = 0
        # self.start_y_pos = 0
        # self.prev_score = 0
        # self.prev_time = 0
        # self.stuck_iter = 0