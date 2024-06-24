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

HyperParameters = {
    "GAME" : "SuperMarioBros3-Nes",
    "CRITIC_LR" : 0.00020184875845596716,
    "ACTOR_LR" : 1.4238752334765434e-06,
    "N_ENVS" : 16,
    # 10,000,000 (desired timesteps) / (N-ENVS * N_STEPS) = N_UPDATES
    "N_UPDATES" : 44643,
    "N_STEPS" : 14,
    "GAMMA" : 0.9748760564790868,
    "LMBDA" : 0.9605167456275167,
    "EPSILON" : 2.582168610117216e-05, # 1e-8 defaut
    "ENT_COEF" : 0.07666385439120696,
    "SAVE_EVERY" : 1000,
    "GRAD_NORM" : 27.295054688617633,
}

Params = {}

