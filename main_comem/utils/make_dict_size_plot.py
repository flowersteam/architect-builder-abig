import argparse
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
from collections import defaultdict, OrderedDict

from alfred.utils.directory_tree import DirectoryTree, get_root
from alfred.utils.misc import select_storage_dirs, robust_seed_aggregate
from alfred.utils.plots import create_fig, bar_chart
from alfred.utils.config import load_dict_from_json
from alfred.utils.recorder import Recorder

from main_comem.main import POSSIBLE_GOALS


def get_make_plots_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_file', type=str, default=None)
    parser.add_argument('--storage_name', type=str, default=None)
    parser.add_argument('--data_pickle_file_name', type=str, default='seed_success_rates.pkl')
    parser.add_argument('--root_dir', default=None, type=str)
    return parser.parse_args()


def gather_seed_scores(storage_dirs, config_filter, data_pickle_file_name):
    seed_scores = OrderedDict()

    experiment_dirs_that_matches_filter = []
    for storage_dir in storage_dirs:
        experiment_dirs = DirectoryTree.get_all_experiments(storage_dir)
        for experiment_dir in experiment_dirs:
            seed_dirs = DirectoryTree.get_all_seeds(experiment_dir)
            expe_config = load_dict_from_json(seed_dirs[0] / 'config.json')

            if passes_filter(expe_config, config_filter):
                experiment_dirs_that_matches_filter.append(experiment_dir)
            else:
                continue

            del expe_config['seed']
            seed_scores[str(expe_config)] = []
            for seed_dir in seed_dirs:
                with open(seed_dir / 'analysis' / data_pickle_file_name, 'rb') as fh:
                    seed_scores[str(expe_config)].append(pickle.load(fh))
                    fh.close()

    return seed_scores, experiment_dirs_that_matches_filter


def passes_filter(config, config_filter):
    return all([v == config[k] for k, v in config_filter.items()])


def sem(x):
    return np.std(x) / (len(x) ** 0.5)


def aggregate_seed_scores(seed_scores):
    expes_dict_of_scores = OrderedDict()
    for expe, list_of_dict_seed_scores in seed_scores.items():
        dict_of_scores_accross_seeds = {}
        for eval_config in list_of_dict_seed_scores[0].keys():
            dict_of_scores_accross_seeds[eval_config] = []
            for seed in list_of_dict_seed_scores:
                dict_of_scores_accross_seeds[eval_config].append(seed[eval_config])

        expe_dict_of_scores = OrderedDict(
            {k: {'mean': np.mean(v), 'sem': sem(v)} for k, v in dict_of_scores_accross_seeds.items()})

        expes_dict_of_scores[expe] = expe_dict_of_scores

    return expes_dict_of_scores


def make_inner_keys_outer_keys(dict_of_dicts):
    old_outer_keys = list(dict_of_dicts.keys())
    old_inner_keys = list(dict_of_dicts[old_outer_keys[0]].keys())

    new_dict_of_dicts = OrderedDict({k: OrderedDict() for k in old_inner_keys})

    for old_out_k in old_outer_keys:
        for old_in_k in old_inner_keys:
            new_dict_of_dicts[old_in_k][old_out_k] = dict_of_dicts[old_out_k][old_in_k].copy()

    return new_dict_of_dicts


GOAL_DICT = {'grasp_object': 'grasp',
             'place_object': 'place',
             'horizontal_line': 'H-line',
             'vertical_line': 'V-line',
             'mixed': 'all-goals'}


def extract_train_goal(s):
    if "'builder_type': 'random'" in s:
        return 'random'

    if "'builder_type': 'saved_from_other_seed_dir'" in s:
        return 'no-goal'

    if "'builder_type': 'saved_initial'" in s:
        if "'make_builder_deterministic': False" in s:
            return 'stochastic'
        elif "'make_builder_deterministic': True":
            return 'deterministic'

    assert 'bw_init_goal' in s

    if "'change_goal': True" in s:
        return GOAL_DICT['mixed']
    for g in POSSIBLE_GOALS:
        if g in s:
            return GOAL_DICT[g]

    raise ValueError


def extract_eval_goal(s):
    if 'goal_override' in s:
        for g in POSSIBLE_GOALS:
            if g in s:
                return GOAL_DICT[g]
    elif 'architect_type' in s:
        assert 'random' in s
        return 'random_archi'

    else:
        raise ValueError


def get_baseline_scores_and_aggregate(baseline_experiment_dir, root_dir):
    baseline_experiment_dir = get_root(root_dir) / baseline_experiment_dir
    seed_dirs = DirectoryTree.get_all_seeds(baseline_experiment_dir)
    seed_baseline_scores = OrderedDict()
    for seed_dir in seed_dirs:
        baseline_dirs = [x for x in (seed_dir / 'baselines').iterdir() if x.is_dir()]
        for baseline_dir in baseline_dirs:
            with open(baseline_dir / 'seed_baseline_success_rates.pkl', 'rb') as fh:
                seed_score_dict = pickle.load(fh)
                fh.close()

            if baseline_dir.name in seed_baseline_scores:
                for k, v in seed_score_dict.items():
                    seed_baseline_scores[baseline_dir.name][k].append(v)
            else:
                seed_baseline_scores[baseline_dir.name] = OrderedDict({k: [v] for k, v in seed_score_dict.items()})

    baseline_aggregated_scores = OrderedDict()
    for baseline, scores_dict in seed_baseline_scores.items():
        baseline_aggregated_scores[baseline] = OrderedDict(
            {eval_key: {'mean': np.mean(eval_scores), 'sem': sem(eval_scores)}
             for eval_key, eval_scores in scores_dict.items()})

    return baseline_aggregated_scores


def clean_keys(dict_of_dicts):
    new_dict_of_dicts = OrderedDict(
        {extract_train_goal(out_k): OrderedDict({extract_eval_goal(in_k): in_v for in_k, in_v in out_v.items()})
         for out_k, out_v in dict_of_dicts.items()})
    return new_dict_of_dicts


if __name__ == "__main__":
    args = get_make_plots_args()

    ## METHODS Perfo-bars #####################################

    storage_dirs = select_storage_dirs(args.from_file, args.storage_name, args.root_dir)

    dict_of_config_filter = {'2': {'bw_init_goal': 'place_object', 'dict_size': 2},
                             '6': {'bw_init_goal': 'place_object', 'dict_size': 6},
                             '10': {'bw_init_goal': 'place_object', 'dict_size': 10},
                             '18': {'bw_init_goal': 'place_object', 'dict_size': 18}}

    seed_scores = OrderedDict()
    experiment_dirs = []

    for dict_size, config_filter in dict_of_config_filter.items():
        s, exp = gather_seed_scores(storage_dirs, config_filter, args.data_pickle_file_name)
        seed_scores[dict_size] = {'place': clean_keys(aggregate_seed_scores(s))['place']['place']}
        experiment_dirs += exp

    expes_dict_of_scores = seed_scores

    n_bars_per_group = len(expes_dict_of_scores.keys())
    group_names = list(expes_dict_of_scores.values())[0].keys()
    n_groups = len(group_names)
    fig, ax = create_fig((1, 1), figsize=(6,5))

    scores_mean = OrderedDict({out_k: OrderedDict({in_k: val_in['mean'] for in_k, val_in in val_out.items()})
                               for out_k, val_out in expes_dict_of_scores.items()})
    scores_err_up = OrderedDict({out_k: OrderedDict({in_k: val_in['sem'] for in_k, val_in in val_out.items()})
                                 for out_k, val_out in expes_dict_of_scores.items()})
    scores_err_down = scores_err_up
    colors = {'18': '#1E77B4', '10': '#5196C6', '6': '#84B6D7', '2': '#B7D5E9'}
    bar_width = bar_chart(ax,
                          scores=scores_mean,
                          err_up=scores_err_up,
                          err_down=scores_err_down,
                          group_names=group_names,
                          colors=colors,
                          ylabel="Mean score (10 seeds)",
                          xlabel="Goal",
                          legend_title="Vocabulary size",
                          fontsize=15.,
                          fontratio=1.2,
                          legend_pos=(0.5, 1.5),
                          make_y_ticks=True,
                          y_ticks=[0., 0.2, 0.4, 0.6, 0.8, 1.],
                          cmap="tab20b"
                          )

    plt.tight_layout()
    for expe_dir in experiment_dirs:
        os.makedirs(expe_dir / 'ood_performance', exist_ok=True)
        fig.savefig(expe_dir / 'ood_performance' / f'performance_dict_size.png', bbox_inches='tight')
