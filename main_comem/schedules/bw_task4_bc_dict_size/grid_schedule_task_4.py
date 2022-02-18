from alfred.utils.misc import check_params_defined_twice

from main_comem import main

# (1) Enter the algorithms to be run for each experiment

ALG_NAMES = ['mcts']

# (2) Enter the task (dataset or rl-environment) to be used for each experiment

TASK_NAMES = ['bw_task4_bc']

# (3) Enter the seeds to be run for each experiment

N_SEEDS = 10
SEEDS = [1 + x for x in range(N_SEEDS)]

# (4) Hyper-parameters

# Here, for each hyperparam, enter the values you want to try in a list.
# All possible combinations will be run as a separate experiment
# Unspecified (or commented out) params will be set to default defines in main.get_training_args

VARIATIONS = {
    'dict_size': [2, 6, 10, 18],
    'n_episodes_per_interaction_step': [600],
    'architect_discount_factor': [0.95],
    'change_goal': [False],
    'bw_init_goal': ['grasp_object'],
    'max_interaction_step': [60],
    'architect_bc_lr': [5e-4],
    'builder_bc_lr': [1e-4],
    'episode_len': [40],
    'architect_reset_optimizer': [True],
    'builder_reset_optimizer': [True],
    'architect_reset_network': [True],
    'builder_reset_network': [True],
    'com_channel_transformation': ['one-hot'],
    'obs_type': ['xy_continuous'],
    'verbose': [False]
}

# Security check to make sure seed, alg_name and task_name are not defined as hyperparams

assert "seed" not in VARIATIONS.keys()
assert "alg_name" not in VARIATIONS.keys()
assert "task_name" not in VARIATIONS.keys()

# Simple security check to make sure every specified parameter is defined only once

check_params_defined_twice(keys=list(VARIATIONS.keys()))


# (5) Function that returns the hyperparameters for the current search

def get_run_args(overwritten_cmd_line):
    return main.get_training_args(overwritten_cmd_line)
