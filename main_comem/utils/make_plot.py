import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re

from alfred.utils.directory_tree import DirectoryTree
from alfred.utils.misc import select_storage_dirs, robust_seed_aggregate
from alfred.utils.config import load_dict_from_json
from alfred.utils.recorder import Recorder


def get_make_plots_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_file', type=str, default=None)
    parser.add_argument('--storage_name', type=str, default=None)
    parser.add_argument('--random_baseline_file', type=str, default=None, help='for bar_plot only')
    parser.add_argument('--mode', type=str, choices=['curve', 'bar_plot'], required=True)
    parser.add_argument('--config_filter', type=eval, default={}, help='A dict of the form {key1:value1,key2:value2} '
                                                                       'IT IS IMPORTANT THAT THERE IS NO WHITESPACES IN'
                                                                       'THE DECLARATION FOR THE PARSING.'
                                                                       'Only runs with config[keyi]==valuei for all i'
                                                                       'will be plotted')
    parser.add_argument('--root_dir', default=None, type=str)
    return parser.parse_args()


def extract_variations(storage_dir):
    variations = load_dict_from_json(filename=str(storage_dir / 'variations.json'))
    experiment_groups = {key: {} for key in variations.keys()}
    for group_key, properties in experiment_groups.items():
        properties['variations'] = variations[group_key]
    real_variation = {key: value for key, value in properties['variations'].items() if
                      len(value) > 1 and key != 'seed'}

    return real_variation


def extract_tag_exp(seed_dirs, real_variations):
    config = load_dict_from_json(filename=str(seed_dirs[0] / 'config.json'))
    return {key: config[key] for key in real_variations.keys()}

def fits_filter(config_filter, config):
    assert all([key in config for key in config_filter]), "you want to filter based on parameters that were not varied"
    return all([config[key] == value for key, value in config_filter.items()])

def compute_random_baseline_score(args):
    # TODO add random baseline compute success rate
    storage_dir = select_storage_dirs(args.from_file, None, args.root_dir)[0]

    experiment = DirectoryTree.get_all_experiments(storage_dir)[0]
    seed_dirs = DirectoryTree.get_all_seeds(experiment)

    averaged_scores = []
    success_rates = []
    for seed_dir in seed_dirs:
        load_dict_from_json(filename=str(seed_dir / 'config.json'))
        loaded_recorder = Recorder.init_from_pickle_file(
            filename=str(seed_dir / 'recorders' / 'train_recorder.pkl'))

        data = 1 / np.array(loaded_recorder.tape['distance_to_optimum'])
        averaged_scores.append(np.mean([p for p in data if p is not None]))
        success_rates.append(np.sum(loaded_recorder.tape['success']) / len(loaded_recorder.tape['success']))

    score_mean = np.mean(averaged_scores)
    score_std = np.std(averaged_scores)
    sr_mean = np.mean(success_rates)
    sr_std = np.std(success_rates)
    return score_mean, score_std, sr_mean, sr_std


def barplot_err(x, y, xerr=None, yerr=None, data=None, **kwargs):
    _data = []
    for _i in data.index:

        _data_i = pd.concat([data.loc[_i:_i]] * 3, ignore_index=True, sort=False)
        _row = data.loc[_i]
        if xerr is not None:
            _data_i[x] = [_row[x] - _row[xerr], _row[x], _row[x] + _row[xerr]]
        if yerr is not None:
            _data_i[y] = [_row[y] - _row[yerr], _row[y], _row[y] + _row[yerr]]
        _data.append(_data_i)

    _data = pd.concat(_data, ignore_index=True, sort=False)

    _ax = sns.barplot(x=x, y=y, data=_data, ci='sd', **kwargs)

    return _ax


def bar_plot_array(args):
    storage_dirs = select_storage_dirs(args.from_file, args.storage_name, args.root_dir)
    aggregated_results = []

    # Single storage dir with two variations (
    if len(storage_dirs) == 1:

        # Reading variations
        storage_dir = storage_dirs[0]
        sorted_experiments = DirectoryTree.get_all_experiments(storage_dir)
        all_seeds_dir = []
        for experiment in sorted_experiments:
            all_seeds_dir = all_seeds_dir + DirectoryTree.get_all_seeds(experiment)

        real_variation = extract_variations(storage_dir)
        if not len(real_variation) == 2:
            real_variation.pop('alg_name')

        for experiment in sorted_experiments:
            seed_dirs = DirectoryTree.get_all_seeds(experiment)
            tag_exp = extract_tag_exp(seed_dirs, real_variation)

            averaged_score = []
            success_rates = []
            for seed_dir in seed_dirs:
                load_dict_from_json(filename=str(seed_dir / 'config.json'))
                loaded_recorder = Recorder.init_from_pickle_file(
                    filename=str(seed_dir / 'recorders' / 'train_recorder.pkl'))

                # checking that parameters in the seed directory are consistent with the variation value !
                config_unique_dict = load_dict_from_json(filename=str(seed_dir / 'config_unique.json'))
                for key in tag_exp:
                    assert config_unique_dict[key] == tag_exp[key]

                assert 'distance_to_optimum' in loaded_recorder.tape.keys() and 'success' in loaded_recorder.tape.keys()

                averaged_score.append(np.mean(1 / np.array(loaded_recorder.tape['distance_to_optimum'])))
                success_rates.append(np.sum(loaded_recorder.tape['success']) / len(loaded_recorder.tape['success']))

            p_mean_over_seed = np.mean(averaged_score)
            p_std_over_seed = np.std(averaged_score)
            sr_mean_over_seed = np.mean(success_rates)
            sr_std_over_seed = np.std(success_rates)
            aggregated_results.append((tag_exp, p_mean_over_seed, p_std_over_seed, sr_mean_over_seed, sr_std_over_seed))

    if len(storage_dirs) > 1:
        for kk, storage_dir in enumerate(storage_dirs):
            sorted_experiments = DirectoryTree.get_all_experiments(storage_dir)
            config = load_dict_from_json(str(storage_dir) + '/experiment1/seed1/config.json')
            real_variation = extract_variations(storage_dir)
            if not len(real_variation) == 2:
                real_variation.pop('alg_name')
            for experiment in sorted_experiments:
                seed_dirs = DirectoryTree.get_all_seeds(experiment)
                tag_exp = extract_tag_exp(seed_dirs, real_variation)

                averaged_score = []
                success_rates = []
                for seed_dir in seed_dirs:
                    load_dict_from_json(filename=str(seed_dir / 'config.json'))
                    loaded_recorder = Recorder.init_from_pickle_file(
                        filename=str(seed_dir / 'recorders' / 'train_recorder.pkl'))

                    # checking that parameters in the seed directory are consistent with the variation value !
                    config_unique_dict = load_dict_from_json(filename=str(seed_dir / 'config_unique.json'))
                    for key in tag_exp:
                        assert config_unique_dict[key] == tag_exp[key]

                    assert 'distance_to_optimum' in loaded_recorder.tape.keys() and 'success' in loaded_recorder.tape.keys()

                    averaged_score.append(np.mean(1 / np.array(loaded_recorder.tape['distance_to_optimum'])))
                    success_rates.append(np.sum(loaded_recorder.tape['success']) / len(loaded_recorder.tape['success']))

                p_mean_over_seed = np.mean(averaged_score)
                p_std_over_seed = np.std(averaged_score)
                sr_mean_over_seed = np.mean(success_rates)
                sr_std_over_seed = np.std(success_rates)
                aggregated_results.append(
                    (tag_exp, p_mean_over_seed, p_std_over_seed, sr_mean_over_seed, sr_std_over_seed))

    columns = list(real_variation.keys()) + ['Score', 'Score std', 'Success Rate', 'Success Rate std']

    df = pd.DataFrame(columns=columns)
    for res in aggregated_results:
        # Reorganizing data for task 3
        if len(real_variation) == 1 and 'task_name' in real_variation.keys() and 'task3_bijective' in real_variation[
            'task_name']:
            dict_df = {**res[0], 'bc': 'bc' if 'bc' in res[0]['task_name'] else 'no_bc', 'Score': res[1],
                       'Score std': res[2], 'Success Rate': res[3],
                       'Success Rate std': res[4]}
            if 'bc' in dict_df['task_name']:
                dict_df['task_name'] = dict_df['task_name'][:-3]

        else:
            dict_df = {**res[0], 'Score': res[1], 'Score std': res[2], 'Success Rate': res[3],
                       'Success Rate std': res[4]}
        df = df.append(dict_df, ignore_index=True)

    # Reorganizing data for task3 bc vs no_bc
    if len(real_variation) == 1 and 'task_name' in real_variation.keys() and 'task3_bijective' in real_variation[
        'task_name']:
        columns.append('bc')
        columns[1], columns[-1] = columns[-1], columns[1]

    # Plotting results
    plt.figure()
    barplot_err(x=columns[0], y="Score", yerr="Score std", hue=columns[1],
                capsize=.2, data=df)
    # plt.axhline(y=score_mean_random, linewidth=1, linestyle='--', color='r', label='random baseline')
    # plt.axhspan(ymin=score_mean_random - score_std_random, ymax=score_mean_random + score_std_random, color='r',
    #             alpha=0.3)
    plt.legend()
    plt.savefig('plots/score_{}.png'.format(re.findall("([^/]+$)", args.from_file)[0][14:-4]))

    plt.figure()
    barplot_err(x=columns[0], y="Success Rate", yerr="Success Rate std", hue=columns[1],
                capsize=.2, data=df)
    # plt.axhline(y=sr_mean_random, linewidth=1, linestyle='--', color='r', label='random baseline')
    # plt.axhspan(ymin=sr_mean_random - sr_std_random, ymax=sr_mean_random + sr_std_random, color='r', alpha=0.3)
    plt.legend()
    plt.savefig('plots/sucess_rate_{}.png'.format(re.findall("([^/]+$)", args.from_file)[0][14:-4]))


def curve_plot_array(config):
    storage_dirs = select_storage_dirs(config.from_file, config.storage_name, config.root_dir)
    aggregated_results = []

    for kk, storage_dir in enumerate(storage_dirs):
        sorted_experiments = DirectoryTree.get_all_experiments(storage_dir)
        real_variation = extract_variations(storage_dir)
        for experiment in sorted_experiments:
            seed_dirs = DirectoryTree.get_all_seeds(experiment)
            tag_exp = extract_tag_exp(seed_dirs, real_variation)

            if not fits_filter(config.config_filter, tag_exp):
                continue

            curve_score_list = []
            curve_sr_list = []
            curve_success_to_optim_list = []
            for seed_dir in seed_dirs:
                loaded_recorder = Recorder.init_from_pickle_file(
                    filename=str(seed_dir / 'recorders' / 'train_recorder.pkl'))

                # checking that parameters in the seed directory are consistent with the variation value !
                config_unique_dict = load_dict_from_json(filename=str(seed_dir / 'config_unique.json'))
                for key in tag_exp:
                    assert config_unique_dict[key] == tag_exp[key]

                assert 'distance_to_optimum' in loaded_recorder.tape.keys() and 'success' in loaded_recorder.tape.keys()

                interaction_steps = np.array(loaded_recorder.tape['interaction_step'])
                distance_to_optimum = 1 / np.array(loaded_recorder.tape['distance_to_optimum'])
                success = np.array(loaded_recorder.tape['success'])
                success_to_optimum = distance_to_optimum * success
                N_step = interaction_steps[-1] + 1
                curve_score, curve_sr, curve_success_to_optim = [], [], []

                for i in range(N_step):
                    curve_score.append(np.mean(distance_to_optimum[interaction_steps == i]))
                    curve_sr.append(np.sum(success[interaction_steps == i]) / len(success[interaction_steps == i]))
                    curve_success_to_optim.append(np.mean(success_to_optimum[interaction_steps == i]))

                curve_score_list.append(curve_score)
                curve_sr_list.append(curve_sr)
                curve_success_to_optim_list.append(curve_success_to_optim)

            p_mean_over_seed = robust_seed_aggregate(curve_score_list, np.mean, np.asarray)
            p_std_over_seed = robust_seed_aggregate(curve_score_list, np.std, np.asarray)

            sr_mean_over_seed = robust_seed_aggregate(curve_sr_list, np.mean, np.asarray)
            sr_std_over_seed = robust_seed_aggregate(curve_sr_list, np.std, np.asarray)

            sto_mean_over_seed = robust_seed_aggregate(curve_success_to_optim_list, np.mean, np.asarray)
            sto_std_over_seed = robust_seed_aggregate(curve_success_to_optim_list, np.std, np.asarray)

            tag_exp.update({'num':seed_dirs[0].parent.name})

            aggregated_results.append(
                (tag_exp, [i for i in range(len(p_mean_over_seed))], p_mean_over_seed, p_std_over_seed,
                 sr_mean_over_seed, sr_std_over_seed, sto_mean_over_seed, sto_std_over_seed))

    plt.figure()
    for res in aggregated_results:
        plt.plot(res[1], res[2], label=str(res[0]))
        up = res[2] + res[3]
        down = res[2] - res[3]
        plt.fill_between(res[1], up, down, alpha=0.2)
    plt.xlabel('interaction_step')
    plt.ylabel('Perf score')
    plt.legend()
    plt.title(f"Filter = {config.config_filter}")

    plt.figure()
    for res in aggregated_results:
        plt.plot(res[1], res[4], label=str(res[0]))
        up = res[4] + res[5]
        down = res[4] - res[5]
        plt.fill_between(res[1], up, down, alpha=0.2)

    plt.xlabel('interaction_step')
    plt.ylabel('Success rate')
    plt.title(f"Filter = {config.config_filter}")
    plt.legend()

    plt.figure()
    for res in aggregated_results:
        plt.plot(res[1], res[6], label=str(res[0]))
        up = res[6] + res[7]
        down = res[6] - res[7]
        plt.fill_between(res[1], up, down, alpha=0.2)

    plt.xlabel('interaction_step')
    plt.ylabel('Success rate x Perf score')
    plt.title(f"Filter = {config.config_filter}")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    args = get_make_plots_args()

    if args.mode == 'bar_plot':
        # args.score_mean_random, args.score_std_random, args.sr_mean_random, args.sr_std_random = compute_random_baseline_score(args)

        bar_plot_array(args)

    elif args.mode == 'curve':
        curve_plot_array(args)

    else:
        raise NotImplementedError
