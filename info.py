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
    "CRITIC_LR" : 1.0029493471189848e-05,
    "ACTOR_LR" : 7.899247199367032e-05,
    "N_ENVS" : 16,
    # 10,000,000 (desired timesteps) / (N-ENVS * N_STEPS) = N_UPDATES
    "N_UPDATES" : 89286,
    "N_STEPS" : 7,
    "GAMMA" : 0.9825276892011053,
    "LMBDA" : 0.9597489250814336,
    "EPSILON" : 1.1847802522966697e-08, # 1e-8 defaut
    "ENT_COEF" : 0.08079379765955479,
    "SAVE_EVERY" : 1000,
    "GRAD_NORM" : 25.816994504967173,
}

Params = {}

