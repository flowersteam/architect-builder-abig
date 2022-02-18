import os
import pickle

from nop import NOP
import logging
import argparse
from pathlib import Path
import warnings

from alfred.utils.config import parse_bool, update_config_unique, parse_log_level, save_config_to_json
from alfred.utils.misc import create_management_objects
from alfred.utils.recorder import Recorder, Aggregator
from alfred.utils.directory_tree import DirectoryTree

import env_comem
import main_comem
from main_comem.utils.misc import save_training_graphs
from main_comem.agents.agent_policy import compute_policy_entropy, compute_accuracy_between_policies, \
    compute_preferred_action_entropy, compute_transitions_probas, compute_MIs

from main_comem.world import make_world


# Setting up alfred
def set_up_alfred():
    DirectoryTree.default_root = "./storage"
    DirectoryTree.git_repos_to_track['main_comem'] = str(Path(main_comem.__file__).parents[1])
    DirectoryTree.git_repos_to_track['env_comem'] = str(Path(env_comem.__file__).parents[1])


set_up_alfred()

POSSIBLE_GOALS = ['grasp_object', 'place_object', 'horizontal_line', 'vertical_line']


def get_training_args(overwritten_cmd_line=None):
    parser = argparse.ArgumentParser()

    ## Alfred's args
    parser.add_argument('--alg_name', type=str, choices=['mcts', 'value_iteration'], help='Overall policy optimization'
                                                                                          ' used')

    parser.add_argument('--task_name', type=str, default='full_random',
                        choices=['full_random',
                                 'task1', 'task1_same_goal',
                                 'task2_mapping', 'task2_mapping_sanity_check', 'task2_mapping_with_goal_info',
                                 'task3_bijective', 'task3_injective', 'task3_stochastic',
                                 'task3_bijective_bc', 'task3_injective_bc', 'task3_stochastic_bc',
                                 'task4', 'task4_archi_random', 'task4_bijective_init', 'task4_bc', 'task4_bc_mlp',
                                 'bw_task4_bc', 'bw_cnn_task4_bc', 'bw_film_task4_bc', 'bw_mlp_emb_task4_bc',
                                 'bw_mlp_attention_task4_bc',
                                 'bw_task4_bc_archi_random',
                                 'none'])

    parser.add_argument('--desc', type=str, default='',
                        help='description of the experiment to be run')

    parser.add_argument('--seed', type=int, default=131214)

    ## World args

    parser.add_argument('--episode_len', type=int, default=20)

    # Env args
    parser.add_argument('--n_objects', type=int, default=3)
    parser.add_argument('--grid_size', type=int, nargs='+', default=(5, 6))
    parser.add_argument('--reward_type', type=str, default='sparse',
                        choices=['manhattan', 'sparse', 'progress'])
    parser.add_argument('--obs_type', type=str, default='xy_discrete',
                        choices=['xy_discrete', 'xy_continuous', 'tile'])
    parser.add_argument('--change_goal', type=parse_bool, default=False)
    parser.add_argument('--bw_init_goal', type=str, choices=POSSIBLE_GOALS,
                        help='For buildworld only')

    # Com args
    parser.add_argument('--dict_size', type=int, default=5)
    parser.add_argument('--com_channel_transformation', type=str, default='identity',
                        choices=['identity', 'one-hot'])
    parser.add_argument('--obs_blender_type', default=None, choices=['obs_dict', None])

    # Agents args
    # architect
    parser.add_argument('--architect_policy_type', type=str, default='none',
                        choices=['random', 'hardcoded_mapping', 'mcts', 'value_iteration', 'none'])
    parser.add_argument('--tilde_builder_type', type=str, default='none',
                        choices=['oracle', 'bc'])
    parser.add_argument('--architect_simulation_type', type=str, default='default')
    parser.add_argument('--architect_budget', type=int, default=100)
    parser.add_argument('--architect_horizon', type=int, default=10)

    parser.add_argument('--architect_keep_tree', type=parse_bool, default=True)
    parser.add_argument('--architect_max_depth', type=int, default=500)
    parser.add_argument('--architect_ucb_cp', type=float, default=2 ** 0.5)
    parser.add_argument('--architect_discount_factor', type=float, default=0.95)

    parser.add_argument('--architect_bc_lr', type=float, default=1e-4)
    parser.add_argument('--architect_bc_max_epoch', type=int, default=1000)
    parser.add_argument('--architect_bc_batch_size', type=int, default=256)
    parser.add_argument('--architect_reset_optimizer', type=parse_bool, default=True)
    parser.add_argument('--architect_reset_network', type=parse_bool, default=False)
    parser.add_argument('--architect_max_wait', type=int, default=300)

    # builder
    parser.add_argument('--builder_policy_type', type=str, default='none',
                        choices=['random', 'mcts', 'value_iteration',
                                 'bijective_mapping', 'injective_mapping', 'stochastic_mapping',
                                 'none'])
    parser.add_argument('--builder_policy_temperature', type=float, default=1.)
    parser.add_argument('--tilde_reward_type', type=str, default='none')
    parser.add_argument('--builder_simulation_type', type=str, default='default')
    parser.add_argument('--builder_budget', type=int, default=100)
    parser.add_argument('--builder_horizon', type=int, default=10)
    parser.add_argument('--builder_keep_tree', type=parse_bool, default=True)
    parser.add_argument('--builder_max_depth', type=int, default=500)
    parser.add_argument('--builder_ucb_cp', type=float, default=2 ** 0.5)
    parser.add_argument('--builder_discount_factor', type=float, default=0.95)
    parser.add_argument('--builder_message_model_type', choices=['softmax', 'uniform'], default='softmax')
    parser.add_argument('--builder_bc_lr', type=float, default=1e-4)
    parser.add_argument('--builder_bc_max_epoch', type=int, default=1000)
    parser.add_argument('--builder_bc_batch_size', type=int, default=256)
    parser.add_argument('--builder_reset_optimizer', type=parse_bool, default=True)
    parser.add_argument('--builder_reset_network', type=parse_bool, default=False)
    parser.add_argument('--builder_max_wait', type=int, default=300)

    ## Management args
    # Static case
    parser.add_argument('--max_episode', type=int, default=None, help="For static-case")
    parser.add_argument('--max_step', type=int, default=None, help="For static-case")
    # Dynamic case
    parser.add_argument('--max_interaction_step', type=int, default=None)
    parser.add_argument('--n_episodes_per_interaction_step', type=int, default=None, help="For dynamic-case")
    parser.add_argument('--n_steps_per_interaction_step', type=int, default=None, help="For dynamic-case")
    parser.add_argument('--make_measurements', type=parse_bool, default=True, help="Make measurements of policies'"
                                                                                   "entropies and similarity")

    parser.add_argument('--verbose', type=parse_bool, default=True)
    parser.add_argument('--sync_wandb', type=parse_bool, default=False)
    parser.add_argument('--use_wandb', type=parse_bool, default=False)
    parser.add_argument('--incremental_save_of_models', type=parse_bool, default=False)
    parser.add_argument('--save_every', type=int, default=float('inf'), help="number of steps between saves")
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--log_level', type=parse_log_level, default=logging.INFO)

    return parser.parse_args(overwritten_cmd_line)


def sanity_check_args(config, logger=None):
    old_dict = config.__dict__.copy()

    # Episode-len for interaction time termination
    if config.episode_len < sum(config.grid_size):
        warnings.warn("episode_len is smaller than the maximum manhattan distance. It may "
                      "be impossible to reach the goal in some episodes", UserWarning)

    # Task-name check

    if 'bw' in config.task_name:
        config.env_type = 'buildworld'
    else:
        config.env_type = 'gridworld'
        config.make_measurement = False

    if config.task_name == 'full_random':
        config.architect_policy_type = 'random'
        config.builder_policy_type = 'random'
        config.architect_policy_args = None
        config.builder_policy_args = None
        config.tilde_reward_model_args = None

        if ((config.max_episode is not None) or (config.max_step is not None)) \
                and \
                ((config.n_episodes_per_interaction_step is None) and (config.n_steps_per_interaction_step is None)):

            config.static_task = True

        elif ((config.n_episodes_per_interaction_step is not None) or (
                config.n_steps_per_interaction_step is not None)) and \
                ((config.max_episode is None) and (config.max_step is None)):

            config.static_task = False

        else:
            raise ValueError("It is not possible to determine if full_random task should be static or not !")


    elif 'task1' in config.task_name:
        config.architect_policy_type = 'random'
        config.builder_policy_type = config.alg_name
        config.architect_policy_args = None
        config.tilde_reward_type = 'oracle'
        config.tilde_reward_model_args = None
        config.builder_policy_args = {'budget': config.builder_budget,
                                      'horizon': config.builder_horizon,
                                      'keep_tree': config.builder_keep_tree,
                                      'max_depth': config.builder_max_depth,
                                      'ucb_cp': config.builder_ucb_cp,
                                      'discount_factor': config.builder_discount_factor,
                                      'message_model_type': config.builder_message_model_type}
        if 'same_goal' in config.task_name:
            config.change_goal = False

        config.static_task = True

    elif 'task2' in config.task_name:

        if 'mapping' in config.task_name:
            config.architect_policy_type = 'hardcoded_mapping'
            config.architect_policy_args = None
            if 'with_goal_info' in config.task_name:
                config.tilde_reward_type = 'action_message_mapping_with_goal_info'
            else:
                config.tilde_reward_type = 'action_message_mapping'
            config.tilde_reward_model_args = None

        else:
            raise NotImplementedError('You need to specify the protocole for task 2')

        config.builder_policy_type = config.alg_name
        config.builder_policy_args = {'budget': config.builder_budget,
                                      'horizon': config.builder_horizon,
                                      'keep_tree': config.builder_keep_tree,
                                      'max_depth': config.builder_max_depth,
                                      'ucb_cp': config.builder_ucb_cp,
                                      'discount_factor': config.builder_discount_factor,
                                      'message_model_type': config.builder_message_model_type}

        config.static_task = True

    elif 'task3' in config.task_name:

        config.architect_policy_type = config.alg_name
        config.architect_policy_args = {'budget': config.architect_budget,
                                        'horizon': config.architect_horizon,
                                        'keep_tree': config.architect_keep_tree,
                                        'max_depth': config.architect_max_depth,
                                        'ucb_cp': config.architect_ucb_cp,
                                        'discount_factor': config.architect_discount_factor,
                                        'message_model_type': config.builder_message_model_type}

        if 'bc' in config.task_name:
            config.tilde_builder_type = 'bc'

            config.static_task = False

        else:
            config.tilde_builder_type = 'oracle'
            config.static_task = True

        if 'bijective' in config.task_name:
            config.builder_policy_type = 'bijective_mapping'

        elif 'injective' in config.task_name:
            config.builder_policy_type = 'injective_mapping'

        elif 'stochastic' in config.task_name:
            config.builder_policy_type = 'stochastic_mapping'

        else:
            raise NotImplementedError

        config.builder_policy_args = None
        config.tilde_reward_model_args = None

    elif 'task4' in config.task_name:

        # architect definition
        if 'archi_random' in config.task_name:
            config.architect_policy_type = 'random'
            config.architect_policy_args = None
            config.tilde_builder_type = "none"
            config.tilde_builder_args = None

        else:
            config.architect_policy_type = config.alg_name
            config.architect_policy_args = {'budget': config.architect_budget,
                                            'horizon': config.architect_horizon,
                                            'keep_tree': config.architect_keep_tree,
                                            'max_depth': config.architect_max_depth,
                                            'ucb_cp': config.architect_ucb_cp,
                                            'discount_factor': config.architect_discount_factor}

            if 'bw' in config.task_name or 'mlp' in config.task_name:
                if 'cnn' in config.task_name:
                    config.tilde_builder_type = 'bc_cnn'
                elif 'film' in config.task_name:
                    config.obs_blender_type = 'obs_dict'
                    config.obs_type = 'tile'
                    config.com_channel_transformation = 'identity'
                    config.tilde_builder_type = 'bc_film'
                elif 'emb' in config.task_name:
                    config.obs_blender_type = 'obs_dict_flat_gw'
                    config.obs_type = 'xy_continuous'
                    config.com_channel_transformation = 'identity'
                    config.tilde_builder_type = 'bc_mlp_emb'
                elif 'attention' in config.task_name:
                    config.obs_blender_type = 'obs_dict_flat_gw'
                    config.obs_type = 'xy_continuous'
                    config.com_channel_transformation = 'identity'
                    config.tilde_builder_type = 'bc_mlp_attention'
                else:
                    config.tilde_builder_type = 'bc_mlp'
                config.tilde_builder_args = {'lr': config.architect_bc_lr,
                                             'max_epoch': config.architect_bc_max_epoch,
                                             'batch_size': config.architect_bc_batch_size,
                                             'reset_optimizer': config.architect_reset_optimizer,
                                             'reset_network': config.architect_reset_network,
                                             'max_wait': config.architect_max_wait}

                config.architect_policy_args.update({'use_heuristic': True})
            else:
                config.tilde_builder_type = 'bc'
                config.tilde_builder_args = {}
                config.architect_policy_args['use_heuristic'] = False

        # builder definition
        if 'bc' in config.task_name:
            config.builder_policy_type = 'bc'
            config.builder_policy_args = None
            config.tilde_reward_model_args = {'temperature': config.builder_policy_temperature}
        else:
            config.builder_policy_type = config.alg_name
            config.builder_policy_args = {'budget': config.builder_budget,
                                          'horizon': config.builder_horizon,
                                          'keep_tree': config.builder_keep_tree,
                                          'max_depth': config.builder_max_depth,
                                          'ucb_cp': config.builder_ucb_cp,
                                          'discount_factor': config.builder_discount_factor,
                                          'message_model_type': config.builder_message_model_type,
                                          'temperature': config.builder_policy_temperature}
        if 'bw' in config.task_name or 'mlp' in config.task_name:
            if 'cnn' in config.task_name:
                config.tilde_reward_type = 'bc_cnn_irl'
            elif 'film' in config.task_name:
                config.tilde_reward_type = 'bc_film_irl'
            elif 'emb' in config.task_name:
                config.tilde_reward_type = 'bc_mlp_emb_irl'
            elif 'attention' in config.task_name:
                config.tilde_reward_type = 'bc_mlp_attention_irl'
            else:
                config.tilde_reward_type = 'bc_mlp_irl'
            config.tilde_reward_model_args.update({'lr': config.builder_bc_lr,
                                                   'max_epoch': config.builder_bc_max_epoch,
                                                   'batch_size': config.builder_bc_batch_size,
                                                   'reset_optimizer': config.builder_reset_optimizer,
                                                   'reset_network': config.builder_reset_network,
                                                   'max_wait': config.builder_max_wait})
        else:
            config.tilde_reward_type = 'bc_irl'

            if 'bijective_init' in config.task_name:
                config.tilde_reward_model_args = {'init': 'bijective'}
            else:
                config.tilde_reward_model_args = {'init': 'stochastic'}

        config.static_task = False

    elif config.task_name == 'none':
        pass

    else:
        raise NotImplementedError

    # To run alfred schedule without running the non-diagonal elements of the grid search
    if config.builder_simulation_type == 'big':
        config.builder_budget = 400
        config.builder_horizon = 20
    elif config.builder_simulation_type == 'very_big':
        config.builder_budget = 800
        config.builder_horizon = 40
    elif config.builder_simulation_type == 'no':
        config.builder_budget = 5
        config.builder_horizon = 0
    elif config.builder_simulation_type == 'default':
        pass
    else:
        raise NotImplementedError

    if config.architect_simulation_type == 'big':
        config.architect_budget = 400
        config.architect_horizon = 20
    elif config.architect_simulation_type == 'no':
        config.architect_budget = 5
        config.architect_horizon = 0
    elif config.architect_simulation_type == 'default':
        pass
    else:
        raise NotImplementedError

    # Step-wise vs. Episode-wise check
    if config.static_task:
        if ((config.max_episode is None) and (config.max_step is None)) \
                or ((config.max_episode is not None) and (config.max_step is not None)):
            raise ValueError("Choose between step-wise or episode-wise run")

    else:
        if ((config.n_episodes_per_interaction_step is None) and (config.n_steps_per_interaction_step is None)) \
                or ((config.n_episodes_per_interaction_step is not None) and (
                config.n_steps_per_interaction_step is not None)):
            raise ValueError("Choose between step-wise or episode-wise interaction-steps")

        if config.max_interaction_step is None:
            raise ValueError("Specify number of interaction steps !")

        if config.n_steps_per_interaction_step is not None:
            raise NotImplementedError

    ## wandb
    if config.sync_wandb == True:
        config.use_wandb = True

    # if we modified the config we redo a sanity check
    if old_dict != config.__dict__:
        config = sanity_check_args(config, logger)
    return config


def main(config, dir_tree=None, pbar="default_pbar", logger=None):
    dir_tree, logger, pbar = create_management_objects(dir_tree=dir_tree, logger=logger, pbar=pbar, config=config)

    # Sanity check config and save it (and update config_unique accordingly)
    config = sanity_check_args(config=config, logger=logger)
    config.experiment_name = str(dir_tree.experiment_dir)  # will be used by wandb
    update_config_unique(config, dir_tree.seed_dir)
    save_config_to_json(config, str(dir_tree.seed_dir / 'config.json'))

    if pbar is not None:
        if config.static_task:
            pbar.total = config.max_step if config.max_step is not None else config.max_episode
        else:
            pbar.total = config.max_interaction_step

    # Importing wandb (or not)
    if not config.sync_wandb:
        os.environ['WANDB_MODE'] = 'dryrun'
        os.environ['WANDB_DISABLE_CODE'] = 'true'

    if config.use_wandb:
        import wandb
        os.environ["WANDB_DIR"] = str(dir_tree.seed_dir.absolute())
        wandb.init(id=dir_tree.get_run_name(), project='comem', entity='irl_la_forge', reinit=True)
        wandb.config.update(config, allow_val_change=True)
    else:
        wandb = NOP()
        wandb_save_dir = None

    # Create world
    env, architect, chnl, obs_blender, builder = make_world(env_type=config.env_type,
                                                            change_goal=config.change_goal,
                                                            bw_init_goal=config.bw_init_goal,
                                                            n_objects=config.n_objects,
                                                            grid_size=config.grid_size,
                                                            reward_type=config.reward_type,
                                                            obs_type=config.obs_type,
                                                            dict_size=config.dict_size,
                                                            com_channel_transformation=config.com_channel_transformation,
                                                            obs_blender_type=config.obs_blender_type,
                                                            architect_policy_type=config.architect_policy_type,
                                                            builder_policy_type=config.builder_policy_type,
                                                            architect_policy_args=config.architect_policy_args,
                                                            builder_policy_args=config.builder_policy_args,
                                                            tilde_builder_type=config.tilde_builder_type,
                                                            tilde_builder_args=config.tilde_builder_args,
                                                            tilde_reward_type=config.tilde_reward_type,
                                                            tilde_reward_model_args=config.tilde_reward_model_args,
                                                            seed=config.seed,
                                                            verbose=config.verbose)

    # initialize loop counters and metrics
    episode = 0
    episode_len = 0
    step = 0
    ret = 0
    steps_on_goal = 0
    reached_time_limit = False
    # init the interaction steps
    interaction_step = 0
    step_per_interaction_step = 0
    episode_per_interaction_step = 0
    is_guiding_phase = False

    # initialize recorder
    to_record = ('return', 'episode_len', 'step', 'success', 'step_per_interaction_step',
                 'episode_per_interaction_step', 'interaction_step', 'episode',
                 'manhattan_distance', 'distance_to_optimum', 'architect_accuracy', 'architect_n_wait',
                 'architect_epoch', 'builder_accuracy', 'builder_n_wait', 'builder_epoch',
                 'architect_tilde_policy_entropy', 'builder_policy_entropy',
                 'architect_tilde_accuracy_to_init', 'builder_accuracy_to_init',
                 'architect_tilde_accuracy_to_previous', 'builder_accuracy_to_previous',
                 'architect_preferred_entropy', 'builder_preferred_entropy',
                 'architect_Isma_p', 'builder_Isma_p',
                 'architect_Isa_p', 'builder_Isa_p',
                 'architect_Ima_p', 'builder_Ima_p',
                 'architect_Isma_pa', 'builder_Isma_pa',
                 'architect_Isa_pa', 'builder_Isa_pa',
                 'architect_Ima_pa', 'builder_Ima_pa'
                 )

    train_recorder = Recorder(to_record)
    train_aggreg = Aggregator()

    # save initial models
    builder.save(dir_tree.recorders_dir / 'builder_init.pyt')
    architect.save(dir_tree.recorders_dir / 'architect_init.pyt')
    builder.save(dir_tree.recorders_dir / 'builder.pyt')
    architect.save(dir_tree.recorders_dir / 'architect.pyt')

    if config.make_measurements:
        measurement_states = env.create_measurement_states(n_states=6000)
        measurement_set = {'states': measurement_states, 'dict_size': chnl.dict_size}
        with open(dir_tree.recorders_dir / 'measurement_set.pkl', 'wb') as fh:
            pickle.dump(measurement_set, fh)
            fh.close()

        initial_builder_policy = builder.init_from_saved(dir_tree.recorders_dir / 'builder_init.pyt').policy
        previous_builder_policy = builder.init_from_saved(dir_tree.recorders_dir / 'builder.pyt').policy

        if not config.tilde_builder_type == 'none':
            initial_architect_tilde_policy = architect.init_from_saved(
                dir_tree.recorders_dir / 'architect_init.pyt').tilde_builder.policy
            previous_architect_tilde_policy = architect.init_from_saved(
                dir_tree.recorders_dir / 'architect.pyt').tilde_builder.policy
            current_tilde_builder_policy = architect.tilde_builder.policy
            architect_obs_blender = architect.obs_blender
            architect_chnl = architect.transform_chnl

        else:
            initial_architect_tilde_policy = None
            previous_architect_tilde_policy = None
            current_tilde_builder_policy = None
            architect_obs_blender = None
            architect_chnl = None

        ## Initial measurements
        measurements = make_measurements(tilde_builder_type=config.tilde_builder_type,
                                         architect_tilde_builder_policy=current_tilde_builder_policy,
                                         measurement_set=measurement_set,
                                         architect_obs_blender=architect_obs_blender,
                                         architect_chnl=architect_chnl,
                                         initial_architect_tilde_policy=initial_architect_tilde_policy,
                                         previous_architect_tilde_policy=previous_architect_tilde_policy,
                                         builder_policy=builder.policy,
                                         builder_obs_blender=builder.obs_blender,
                                         builder_chnl=chnl,
                                         initial_builder_policy=initial_builder_policy,
                                         previous_builder_policy=previous_builder_policy)

        measurements.update({'interaction_step': interaction_step})
        train_recorder.write_to_tape(measurements)

    # init gridworld
    gw_obs, gw_goal = env.reset()
    architect.update_policy()
    builder.update_policy()
    initial_distance = env._compute_n_steps_optim(gw_obs)
    env.render(everything=True)

    # interaction loop
    while True:

        ### STATIC TASKS
        if config.static_task:
            # Protocols are pre-defined and agents are not updated

            if env.done(gw_obs):
                steps_on_goal += 1

            # architect step

            message = architect.act(gw_obs)

            # communication channel step
            chnl.send(message)
            chnl_obs = chnl.read()
            chnl.render(config.verbose)
            blended_obs = obs_blender.blend(gw_obs=gw_obs, chnl_obs=chnl_obs)

            # builder step
            if config.task_name == 'task2_mapping_sanity_check':
                action = message
            else:
                action = builder.act(blended_obs)

            # gridworld step
            next_gw_obs, reward, _, info = env.step(action)

            env.render()

            gw_obs = next_gw_obs
            ret += reward
            step += 1
            episode_len += 1

            train_aggreg.record('manhattan_distance', info['manhattan_distance'])

            if episode_len >= config.episode_len:
                reached_time_limit = True

            if reached_time_limit:
                if steps_on_goal > 0:
                    success = 1
                else:
                    success = 0
                new_recordings = {'return': ret, 'episode_len': episode_len,
                                  'success': success,
                                  'step': step, 'episode': episode,
                                  'distance_to_optimum': (episode_len - steps_on_goal) / (initial_distance + 1e-4)}
                new_recordings.update(train_aggreg.pop_all_means())

                train_recorder.write_to_tape(new_recordings)
                wandb.log(new_recordings, step=step)
                gw_obs, gw_goal = env.reset()
                architect.reset_policy()
                builder.reset_policy()

                architect.update_policy()  # we have to update the architect policy to account for the new goal location
                if builder.tilde_reward_type in ["oracle", "action_message_mapping_with_goal_info"]:
                    if not config.change_goal:
                        builder.update_policy()

                initial_distance = env._compute_n_steps_optim(gw_obs)
                env.render(everything=True)
                ret = 0
                episode_len = 0
                steps_on_goal = 0
                reached_time_limit = False
                episode += 1

                if pbar is not None and config.max_step is None:
                    pbar.update()

            if pbar is not None and config.max_step is not None:
                pbar.update()

            if step % config.save_every == 0:
                new_recordings = {'return': ret, 'episode_len': episode_len,
                                  'step': step, 'episode': episode}
                new_recordings.update(train_aggreg.pop_all_means())

                train_recorder.write_to_tape(new_recordings)
                wandb.log(new_recordings, step=step)

                train_recorder.save(dir_tree.recorders_dir / 'train_recorder.pkl')
                save_training_graphs(train_recorder, dir_tree.seed_dir)

            if (config.max_episode is not None) and episode >= config.max_episode:
                break

            if (config.max_step is not None) and step >= config.max_step:
                break

        ### DYNAMIC TASKS
        else:
            # Agents are updated through interaction steps, we alternate between:
            #   - Guiding steps: architect exploits its builder-model to get it to
            #   destination while the builder collects guidance data. Also corresponds to the phase where
            #   we monitor the performance of the dyiad. The builder reward model is updated with the guidance data
            #   - Modeling steps: new data is generated and the architect's builder-model is updated

            ## GUIDING PHASE
            if is_guiding_phase:
                if env.done(gw_obs):
                    steps_on_goal += 1

                # architect step
                message = architect.act(gw_obs)

                # communication channel step
                chnl.send(message)
                chnl_obs = chnl.read()
                chnl.render(config.verbose)
                blended_obs = obs_blender.blend(gw_obs=gw_obs, chnl_obs=chnl_obs)

                # builder step
                action = builder.act(blended_obs)

                # gridworld step
                next_gw_obs, reward, _, info = env.step(action)

                builder.store(gw_obs=gw_obs, chnl_obs=chnl_obs, action=action)

                env.render(everything=True)

                gw_obs = next_gw_obs
                ret += reward
                step_per_interaction_step += 1
                episode_len += 1

                # train_aggreg.record('manhattan_distance', info['manhattan_distance'])

                if episode_len >= config.episode_len:
                    reached_time_limit = True

                if reached_time_limit:
                    if steps_on_goal > 0:
                        success = 1
                    else:
                        success = 0
                    new_recordings = {'return': ret, 'episode_len': episode_len,
                                      'success': success,
                                      'episode_per_interaction_step': episode_per_interaction_step,
                                      'distance_to_optimum': (episode_len - steps_on_goal) / (initial_distance + 1e-4)}
                    train_aggreg.update(new_recordings)

                    gw_obs, gw_goal = env.reset()
                    architect.reset_policy()
                    builder.reset_policy()

                    architect.update_policy()  # we have to update the architect policy to account for the new goal location
                    initial_distance = env._compute_n_steps_optim(gw_obs)
                    env.render(everything=True)
                    ret = 0
                    episode_len = 0
                    steps_on_goal = 0
                    reached_time_limit = False
                    episode_per_interaction_step += 1

                    logger.info(
                        f"ITERATION {interaction_step} -- GUIDING-PHASE: {episode_per_interaction_step}/{config.n_episodes_per_interaction_step // 2}")

                # we have done enough guiding episodes
                if config.n_episodes_per_interaction_step is not None \
                        and episode_per_interaction_step == int(config.n_episodes_per_interaction_step / 2):

                    interaction_step += 1
                    episode_per_interaction_step = 0

                    # we switch guiding to false because builder has changed and we must update the architect's
                    # builder-model

                    is_guiding_phase = False

                    # we update the builder's tilde reward and erase its memory
                    b_acc, b_n_wait, b_epoch = builder.update_irl()
                    builder.update_policy()  # we have to update the policy because the reward has changed
                    builder.buffer.clear()

                    recordings = train_aggreg.pop_all_means()
                    recordings.update(
                        {'interaction_step': interaction_step, 'builder_accuracy': b_acc, 'builder_n_wait': b_n_wait,
                         'builder_epoch': b_epoch,
                         'architect_accuracy': a_acc, 'architect_n_wait': a_n_wait,
                         'architect_epoch': a_epoch})

                    if config.make_measurements:

                        previous_builder_policy = builder.init_from_saved(dir_tree.recorders_dir / 'builder.pyt').policy
                        if not config.tilde_builder_type == 'none':
                            previous_architect_tilde_policy = architect.init_from_saved(
                                dir_tree.recorders_dir / 'architect.pyt').tilde_builder.policy
                            current_tilde_builder_policy = architect.tilde_builder.policy

                        measurements = make_measurements(tilde_builder_type=config.tilde_builder_type,
                                                         architect_tilde_builder_policy=current_tilde_builder_policy,
                                                         measurement_set=measurement_set,
                                                         architect_obs_blender=architect_obs_blender,
                                                         architect_chnl=architect_chnl,
                                                         initial_architect_tilde_policy=initial_architect_tilde_policy,
                                                         previous_architect_tilde_policy=previous_architect_tilde_policy,
                                                         builder_policy=builder.policy,
                                                         builder_obs_blender=builder.obs_blender,
                                                         builder_chnl=chnl,
                                                         initial_builder_policy=initial_builder_policy,
                                                         previous_builder_policy=previous_builder_policy)

                        recordings.update(measurements)

                    train_recorder.write_to_tape(recordings)

                    # Save recorder add the end of each interaction step
                    # we save the recorders and the models after the guiding phase because the performance
                    # is measured on the guiding phase

                    train_recorder.save(dir_tree.recorders_dir / 'train_recorder.pkl')
                    save_training_graphs(train_recorder, dir_tree.seed_dir)
                    builder.save(dir_tree.recorders_dir / 'builder.pyt')
                    architect.save(dir_tree.recorders_dir / 'architect.pyt')

                    if config.incremental_save_of_models:
                        builder.save(dir_tree.incrementals_dir / f'builder_{interaction_step}.pyt')
                        architect.save(dir_tree.incrementals_dir / f'architect_{interaction_step}.pyt')

                    if pbar is not None and config.max_interaction_step is not None:
                        pbar.update()

                # if we have done enough interaction-steps we stop
                if (config.max_interaction_step is not None) and interaction_step >= config.max_interaction_step:
                    break

            ## MODELLING PHASE
            else:
                # architect RANDOM step
                message = architect.act_space.sample()

                # communication channel step
                chnl.send(message)
                chnl_obs = chnl.read()
                blended_obs = obs_blender.blend(gw_obs=gw_obs, chnl_obs=chnl_obs)

                # builder step
                action = builder.act(blended_obs)

                # gridworld step
                next_gw_obs, _, _, _ = env.step(action)

                architect.store(gw_obs=gw_obs, message=message, action=action)

                gw_obs = next_gw_obs
                step_per_interaction_step += 1
                episode_len += 1

                if episode_len >= config.episode_len:
                    reached_time_limit = True

                if reached_time_limit:
                    gw_obs, gw_goal = env.reset()
                    builder.reset_policy()  # only builder here since architect uses random messages(for the moment)

                    episode_len = 0
                    reached_time_limit = False
                    episode_per_interaction_step += 1

                    logger.info(
                        f"ITERATION {interaction_step} -- MODELLING-PHASE: {episode_per_interaction_step}/{config.n_episodes_per_interaction_step // 2}")

                # we have done enough modeling episodes
                if config.n_episodes_per_interaction_step is not None \
                        and episode_per_interaction_step == int(config.n_episodes_per_interaction_step / 2):

                    # we update the architect builder-model and erase its memory
                    a_acc, a_n_wait, a_epoch = architect.update_bc()
                    architect.buffer.clear()
                    architect.update_policy()  # we update the architect's policy to account for the new builder's model

                    episode_per_interaction_step = 0

                    # we switch to guiding because now models are up-to-date

                    is_guiding_phase = True
                    env.render(everything=True)

                    if pbar is not None and config.max_interaction_step is not None:
                        pbar.update()

    # Final save
    train_recorder.save(dir_tree.recorders_dir / 'train_recorder.pkl')
    save_training_graphs(train_recorder, dir_tree.seed_dir)


def make_measurements(tilde_builder_type, architect_tilde_builder_policy, measurement_set, architect_obs_blender,
                      architect_chnl, initial_architect_tilde_policy, previous_architect_tilde_policy, builder_policy,
                      builder_obs_blender, builder_chnl,
                      initial_builder_policy, previous_builder_policy):
    if not tilde_builder_type == 'none':
        architect_tilde_policy_entropy = compute_policy_entropy(architect_tilde_builder_policy,
                                                                measurement_set,
                                                                architect_obs_blender,
                                                                architect_chnl)

        architect_tilde_accuracy_to_init = compute_accuracy_between_policies(
            reference_policy=initial_architect_tilde_policy,
            policy=architect_tilde_builder_policy,
            measurement_set=measurement_set,
            obs_blender=architect_obs_blender,
            chnl=architect_chnl)

        architect_tilde_accuracy_to_previous = compute_accuracy_between_policies(
            reference_policy=previous_architect_tilde_policy,
            policy=architect_tilde_builder_policy,
            measurement_set=measurement_set,
            obs_blender=architect_obs_blender,
            chnl=architect_chnl)

        nums_archi, Psma_archi, totals_archi, P_m_bar_s_archi = compute_transitions_probas(
            architect_tilde_builder_policy,
            measurement_set,
            architect_obs_blender,
            architect_chnl)
        architect_preferred_entropy = compute_preferred_action_entropy(nums_archi, Psma_archi, totals_archi)

        architect_Isma_p, architect_Isma_pa, architect_Isa_p, architect_Isa_pa, architect_Ima_p, architect_Ima_pa \
            = compute_MIs(nums_archi, Psma_archi, P_m_bar_s_archi, totals_archi)
    else:
        architect_tilde_accuracy_to_init = -1.
        architect_tilde_policy_entropy = -1.
        architect_tilde_accuracy_to_previous = -1.
        architect_preferred_entropy = -1.
        architect_Isma_p = -1.
        architect_Isa_p = -1.
        architect_Ima_p = -1.
        architect_Isma_pa = -1.
        architect_Isa_pa = -1.
        architect_Ima_pa = -1.

    builder_policy_entropy = compute_policy_entropy(builder_policy,
                                                    measurement_set,
                                                    builder_obs_blender,
                                                    builder_chnl)

    builder_accuracy_to_init = compute_accuracy_between_policies(
        reference_policy=initial_builder_policy,
        policy=builder_policy,
        measurement_set=measurement_set,
        obs_blender=builder_obs_blender,
        chnl=builder_chnl)

    builder_accuracy_to_previous = compute_accuracy_between_policies(
        reference_policy=previous_builder_policy,
        policy=builder_policy,
        measurement_set=measurement_set,
        obs_blender=builder_obs_blender,
        chnl=builder_chnl)

    nums_builder, Psma_builder, totals_builder, P_m_bar_s_builder = compute_transitions_probas(builder_policy,
                                                                                               measurement_set,
                                                                                               builder_obs_blender,
                                                                                               builder_chnl)

    builder_preferred_entropy = compute_preferred_action_entropy(nums_builder, Psma_builder, totals_builder)

    builder_Isma_p, builder_Isma_pa, builder_Isa_p, builder_Isa_pa, builder_Ima_p, builder_Ima_pa = compute_MIs(
        nums_builder, Psma_builder, P_m_bar_s_builder, totals_builder)

    return {'architect_tilde_policy_entropy': architect_tilde_policy_entropy,
            'builder_policy_entropy': builder_policy_entropy,
            'architect_tilde_accuracy_to_init': architect_tilde_accuracy_to_init,
            'builder_accuracy_to_init': builder_accuracy_to_init,
            'architect_tilde_accuracy_to_previous': architect_tilde_accuracy_to_previous,
            'builder_accuracy_to_previous': builder_accuracy_to_previous,
            'architect_preferred_entropy': architect_preferred_entropy,
            'builder_preferred_entropy': builder_preferred_entropy,
            'architect_Isma_p': architect_Isma_p,
            'builder_Isma_p': builder_Isma_p,
            'architect_Isa_p': architect_Isa_p,
            'builder_Isa_p': builder_Isa_p,
            'architect_Ima_p': architect_Ima_p,
            'builder_Ima_p': builder_Ima_p,
            'architect_Isma_pa': architect_Isma_pa,
            'builder_Isma_pa': builder_Isma_pa,
            'architect_Isa_pa': architect_Isa_pa,
            'builder_Isa_pa': builder_Isa_pa,
            'architect_Ima_pa': architect_Ima_pa,
            'builder_Ima_pa': builder_Ima_pa}


if __name__ == "__main__":
    set_up_alfred()
    config = get_training_args()
    main(config)
