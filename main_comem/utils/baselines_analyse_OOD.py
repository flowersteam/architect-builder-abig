import argparse
import os
import pickle
from collections import OrderedDict

from alfred.utils.config import parse_bool
from alfred.utils.misc import select_storage_dirs
from alfred.utils.directory_tree import DirectoryTree, get_root

from main_comem.evaluate import get_eval_args, evaluate


def get_analyse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--reference_experiment_dir', type=str)
    parser.add_argument('--self_imitating_experiment_dir', type=str)
    parser.add_argument('--re_run_if_exists', type=parse_bool, default=False,
                        help="Whether to re-compute seed_scores if 'seed_scores.pkl' already exists")
    parser.add_argument('--n_eval_runs', type=int, default=300,
                        help="Only used if performance_metric=='evaluation_runs'")

    parser.add_argument('--root_dir', default=None, type=str)

    return parser.parse_args()


def baselines_OOD_analysis(reference_experiment_dir, self_imitating_experiment_dir, baseline_def_configs,
                           config_override_list, n_eval_runs):
    config = get_eval_args("")
    config.verbose = False
    config.print_builder_policy = False
    config.max_episode = n_eval_runs
    config.root_dir = ''

    seed_dirs = DirectoryTree.get_all_seeds(reference_experiment_dir)

    for i, seed_dir in enumerate(seed_dirs):

        os.makedirs(seed_dir / 'baselines', exist_ok=True)

        for baseline in baseline_def_configs:

            if (seed_dir / 'baselines' / str(baseline)).exists():
                print(f'{seed_dir} --- SKIPPING --- {baseline}')
                continue

            else:
                os.makedirs(seed_dir / 'baselines' / str(baseline), exist_ok=False)

            seed_scores = OrderedDict()

            for eval_config in config_override_list:
                config = get_eval_args("")
                config.verbose = False
                config.print_builder_policy = False
                config.max_episode = n_eval_runs
                config.root_dir = ''
                config.seed_dir = seed_dir

                if 'saved_from_other_seed_dir' in str(baseline):
                    config.__dict__.update({'other_seed_dir_to_load_builder_from':
                                                DirectoryTree.get_all_seeds(self_imitating_experiment_dir)[i]})

                config.__dict__.update(baseline)
                config.__dict__.update(eval_config)

                seed_scores[str(eval_config)] = evaluate(config)

            with open(seed_dir / 'baselines' / str(baseline) / 'seed_baseline_success_rates.pkl', 'wb') as fh:
                pickle.dump(seed_scores, fh)
                fh.close()


if __name__ == "__main__":
    args = get_analyse_args()

    reference_experiment_dir = get_root(args.root_dir) / args.reference_experiment_dir
    self_imitating_experiment_dir = get_root(args.root_dir) / args.self_imitating_experiment_dir

    baseline_def_configs = [{'builder_type': 'random'},
                            {'builder_type': 'saved_from_other_seed_dir', 'architect_type': 'builder_copy'},
                            {'builder_type': 'saved_initial', 'architect_type': 'builder_copy',
                             'make_builder_deterministic': False},
                            {'builder_type': 'saved_initial', 'architect_type': 'builder_copy',
                             'make_builder_deterministic': True}]

    # first we analyse the baselines architect-builder pair on different fixed goals
    config_static = {'change_goal': False}
    config_varying = {'goal_override': ['grasp_object', 'place_object', 'horizontal_line', 'vertical_line']}
    config_override_list = []
    for goal_type in config_varying['goal_override']:
        config_temp = config_static.copy()
        config_temp['goal_override'] = goal_type
        config_override_list.append(config_temp)

    # finally we want the baseline builder but with a random architect on its training goal
    config_override_list.append({'architect_type': 'random', 'change_goal': False})

    baselines_OOD_analysis(reference_experiment_dir, self_imitating_experiment_dir, baseline_def_configs,
                           config_override_list, args.n_eval_runs)
