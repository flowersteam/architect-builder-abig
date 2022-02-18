import argparse
import os
import pickle

from alfred.utils.config import parse_bool
from alfred.utils.misc import select_storage_dirs
from alfred.utils.directory_tree import DirectoryTree

from main_comem.evaluate import get_eval_args, evaluate


def get_analyse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--from_file', type=str, default=None)
    parser.add_argument('--storage_names', type=str, nargs='+', default=None)

    parser.add_argument('--re_run_if_exists', type=parse_bool, default=False,
                        help="Whether to re-compute seed_scores if 'seed_scores.pkl' already exists")
    parser.add_argument('--n_eval_runs', type=int, default=300,
                        help="Only used if performance_metric=='evaluation_runs'")

    parser.add_argument('--root_dir', default=None, type=str)

    return parser.parse_args()


def protocol_OOD_analyse(from_file, storage_names, root_dir, config_override_list, n_eval_runs):
    storage_dirs = select_storage_dirs(from_file, storage_names, root_dir)

    for k, storage_dir in enumerate(storage_dirs):
        print('++++++++++++++++++++++++++++++++++++')
        print(f'{k}/{len(storage_dirs)} CURRENT STORAGE DIR --- {storage_dir}')

        experiment_dirs = DirectoryTree.get_all_experiments(storage_dir)

        for j, experiment_dir in enumerate(experiment_dirs):
            print('==========================')
            print(f'{j}/{len(experiment_dirs)} CURRENT EXPE DIR --- {experiment_dir}')

            seed_dirs = DirectoryTree.get_all_seeds(experiment_dir)

            for i, seed_dir in enumerate(seed_dirs):
                print('-------------------------')
                print(f'{i}/{len(seed_dirs)} CURRENT SEED DIR --- {seed_dir}')

                if (seed_dir / 'analysis').exists() and not args.re_run_if_exists:
                    print(f'PASSED because already computed')
                    continue

                os.makedirs(seed_dir / 'analysis', exist_ok=True)

                seed_perf = {}

                for l, config_variation in enumerate(config_override_list):
                    config = get_eval_args("")
                    config.verbose = False
                    config.print_builder_policy = False
                    config.max_episode = n_eval_runs
                    config.root_dir = ''
                    config.seed_dir = seed_dir

                    config.__dict__.update(config_variation)
                    print(f'{l}/{len(config_override_list)} OVERRIDING CONFIG -- {config.__dict__}')
                    seed_perf[str(config_variation)] = evaluate(config)

                with open(seed_dir / 'analysis' / 'seed_success_rates.pkl', 'wb') as fh:
                    pickle.dump(seed_perf, fh)
                    fh.close()


def combine_params_variation(to_vary):
    # to_vary = {'architect_type': ['saved', 'random'], 'same_goal': [True, False]}
    combinations = [to_vary.copy()]
    for combining_key, combining_vals in to_vary.items():
        expanded_combinations = []
        for comb in combinations:
            for val in combining_vals:
                exp_comb = comb.copy()
                exp_comb.update({combining_key: val})
                expanded_combinations.append(exp_comb)
        combinations = expanded_combinations
    return combinations


if __name__ == "__main__":
    args = get_analyse_args()

    # first we analyse the trained architect-builder pair on different fixed goals
    config_static = {'architect_type': 'saved', 'change_goal': False, 'episode_len_override':60}
    # config_varying = {'goal_override': ['grasp_object', 'place_object', 'horizontal_line', 'vertical_line']}
    config_varying = {'goal_override': ['make_shape']}


    config_override_list = []
    for goal_type in config_varying['goal_override']:
        config_temp = config_static.copy()
        config_temp['goal_override'] = goal_type
        config_override_list.append(config_temp)

    # finally we want the trained builder but with a random architect on its training goal
    config_override_list.append({'architect_type': 'random', 'change_goal': False, 'episode_len_override':60})

    protocol_OOD_analyse(args.from_file, args.storage_names, args.root_dir, config_override_list, args.n_eval_runs)
