import argparse
import numpy as np

from alfred.utils.directory_tree import get_root
from alfred.utils.config import load_config_from_json, parse_bool

from main_comem.world import make_world
from main_comem.agents.agent_policy import Policy


def get_eval_args(overwritten_cmd_line=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_dir', type=str)
    parser.add_argument('--builder_type', type=str, choices=['init', 'saved', 'saved_initial', 'saved_from_other_seed_dir'],
                        default='saved')
    parser.add_argument('--other_seed_dir_to_load_builder_from', type=str, default=None)
    parser.add_argument('--make_builder_deterministic', type=parse_bool, default=False)
    parser.add_argument('--architect_type', type=str, choices=['builder_copy', 'init', 'saved', 'random', 'controlled'],
                        default='saved', help='controllable policy doesnt work on pycharm, launch it from shell')
    parser.add_argument('--change_goal', type=str, choices=['True', 'False', 'saved'], default='saved')
    parser.add_argument('--goal_override', type=str, choices=['grasp_object', 'horizontal_line', 'place_object',
                                                              'make_shape'],
                        default=None)
    parser.add_argument('--seed_override', type=int, default=None)
    parser.add_argument('--max_episode', type=int, default=10)
    parser.add_argument('--episode_len_override', type=int, default=None)
    parser.add_argument('--verbose', type=parse_bool, default=True)
    parser.add_argument('--print_builder_policy', type=parse_bool, default=True)
    parser.add_argument('--root_dir', type=str, default=None)
    return parser.parse_args(overwritten_cmd_line)


def evaluate(args):
    run_dir = get_root(args.root_dir) / args.seed_dir
    config = load_config_from_json(run_dir / 'config.json')
    print(f'SAVED CONFIG -- {config.__dict__}')
    print(f'EVAL CONFIG -- {args.__dict__}')

    if args.change_goal == 'saved':
        change_goal = config.change_goal
    elif args.change_goal in ['True', True]:
        change_goal = True
    elif args.change_goal in ['False', False]:
        change_goal = False
    else:
        raise NotImplementedError

    if args.goal_override is not None:
        bw_init_goal = args.goal_override
    else:
        bw_init_goal = config.bw_init_goal

    if args.episode_len_override is not None:
        max_episode_len = args.episode_len_override
    else:
        max_episode_len = config.episode_len

    # Create world
    gw, architect, chnl, obs_blender, builder = make_world(grid_size=config.grid_size,
                                                           reward_type=config.reward_type,
                                                           obs_type=config.obs_type,
                                                           dict_size=config.dict_size,
                                                           com_channel_transformation=config.com_channel_transformation,
                                                           architect_policy_type=config.architect_policy_type,
                                                           builder_policy_type=config.builder_policy_type,
                                                           architect_policy_args=config.architect_policy_args,
                                                           builder_policy_args=config.builder_policy_args,
                                                           tilde_builder_type=config.tilde_builder_type,
                                                           tilde_reward_type=config.tilde_reward_type,
                                                           tilde_reward_model_args=config.tilde_reward_model_args,
                                                           bw_init_goal=bw_init_goal,
                                                           change_goal=change_goal,
                                                           env_type=config.env_type,
                                                           tilde_builder_args=config.tilde_builder_args,
                                                           seed=config.seed if args.seed_override is None else args.seed_override,
                                                           obs_blender_type=config.obs_blender_type,
                                                           verbose=args.verbose)

    # init gridworld to be able to compare with saved one
    gw_obs, gw_goal = gw.reset()
    gw.render(everything=True)

    if args.builder_type == 'saved':
        builder = builder.init_from_saved(run_dir / 'recorders' / 'builder.pyt',
                                          obs_blender=obs_blender, gridworld_model=gw)
    elif args.builder_type == 'saved_initial':
        builder = builder.init_from_saved(run_dir / 'recorders' / 'builder_init.pyt',
                                          obs_blender=obs_blender, gridworld_model=gw)
    elif args.builder_type == 'init':
        pass
    elif args.builder_type == 'saved_from_other_seed_dir':
        assert args.other_seed_dir_to_load_builder_from is not None
        other_run_dir = get_root(args.root_dir) / args.other_seed_dir_to_load_builder_from

        builder = builder.init_from_saved(other_run_dir / 'recorders' / 'builder.pyt',
                                          obs_blender=obs_blender, gridworld_model=gw)
    elif args.builder_type == 'random':
        pass
    else:
        raise NotImplementedError

    if args.architect_type in ['saved', 'builder_copy']:

        # # todo: what follows is an ugly fix because we forgot to save some variable in init_dict
        # from main_comem.utils.ml import load_checkpoint, save_checkpoint
        # architect_file_path = run_dir / 'recorders' / 'architect.pkl'
        # saved_architect = load_checkpoint(architect_file_path)
        # if 'tilde_builder_args' not in saved_architect['init_dict']:
        #     saved_architect['init_dict']['tilde_builder_args'] = config.tilde_builder_args
        # save_checkpoint(saved_architect, architect_file_path)

        architect = architect.init_from_saved(run_dir / 'recorders' / 'architect.pyt', gridworld_model=gw,
                                              change_goal=change_goal, config_change_goal=config.change_goal)
        if args.architect_type == 'builder_copy':
            architect.tilde_builder.policy = builder.policy

    elif args.architect_type in ['init', 'random']:
        pass
    elif args.architect_type == 'controlled':
        architect = InteractivePolicy(config.dict_size)
    else:
        raise NotImplementedError

    if args.make_builder_deterministic:
        builder.policy.make_deterministic()

    # initialize loop counters and metrics
    episode = 0
    episode_len = 0
    step = 0
    ret = 0
    steps_on_goal = 0
    reached_time_limit = False
    success_list = []

    # init agents
    architect.update_policy()
    builder.update_policy()

    if args.print_builder_policy and builder.policy_type == 'value_iteration':
        builder.policy.print_most_likely_action()

    while True:
        if gw.done(gw_obs):
            steps_on_goal += 1

        # architect step

        if args.architect_type == 'random':
            message = chnl.send_space.sample()
        elif args.architect_type in ['saved', 'init', 'controlled', 'builder_copy']:
            message = architect.act(gw_obs)
        else:
            raise NotImplementedError

        # communication channel step
        chnl.send(message)
        chnl_obs = chnl.read()
        chnl.render(args.verbose)
        blended_obs = obs_blender.blend(gw_obs=gw_obs, chnl_obs=chnl_obs)

        # builder step
        if config.task_name == 'task2_mapping_sanity_check':
            action = message
        elif args.builder_type == 'random':
            action = gw.action_space.sample()
        else:
            action = builder.act(blended_obs)

        # gridworld step
        next_gw_obs, reward, _, info = gw.step(action)

        gw.render()

        gw_obs = next_gw_obs
        ret += reward
        step += 1
        episode_len += 1

        if episode_len >= max_episode_len:
            reached_time_limit = True

        if reached_time_limit:

            if steps_on_goal > 0:
                success = 1
            else:
                success = 0

            success_list.append(success)

            gw_obs, gw_goal = gw.reset()
            architect.reset_policy()
            builder.reset_policy()

            architect.update_policy()  # we have to update the architect policy to account for the new goal location

            if builder.tilde_reward_type in ["oracle", "action_message_mapping_with_goal_info"]:
                builder.update_policy()

            gw.render(everything=config.env_type == 'gridworld')
            ret = 0
            episode_len = 0
            steps_on_goal = 0
            reached_time_limit = False
            episode += 1

        if (args.max_episode is not None) and episode >= args.max_episode:
            return np.mean(success_list)


class InteractivePolicy(Policy):
    def __init__(self, dict_size):
        import readchar
        self.dict_size = dict_size
        self.readchar = readchar

    def act(self, *args, **kwargs):
        while True:
            key = self.readchar.readkey()
            word = int(key)
            if word >= 0 and word < self.dict_size:
                return word
            print("word out of vocabulary")

    def reset_policy(self, *args, **kwargs):
        pass

    def update_policy(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    args = get_eval_args()
    evaluate(args)
