import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import pickle
from collections import defaultdict

from alfred.utils.directory_tree import DirectoryTree
from alfred.utils.misc import select_storage_dirs, robust_seed_aggregate
from alfred.utils.plots import create_fig, plot_curves
from alfred.utils.config import load_dict_from_json
from alfred.utils.recorder import Recorder


def get_make_plots_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_file', type=str, default=None)
    parser.add_argument('--storage_name', type=str, default=None)
    parser.add_argument('--curve_group_list', type=eval, default=DEFAULT_CURVE_GROUP_LIST)
    parser.add_argument('--x_metric', type=str, default='interaction_step')
    parser.add_argument('--error_bar_type', type=str, choices=['std', 'sem'], default='std')
    parser.add_argument('--data_pickle_file_name', type=str, default='train_recorder.pkl')
    parser.add_argument('--root_dir', default=None, type=str)
    return parser.parse_args()


DEFAULT_CURVE_GROUP_LIST = [
    ('scores', (['architect_accuracy', 'builder_accuracy', 'success', 'builder_accuracy_to_init',
                                      'builder_accuracy_to_previous'], 'interaction_step')),
    ('entropies', (['builder_policy_entropy', 'builder_preferred_entropy'], 'interaction_step')),
    ('MI', (['builder_Isma_p', 'builder_Isa_p', 'builder_Ima_p', 'builder_Isma_pa', 'builder_Isa_pa', 'builder_Ima_pa'],
            'interaction_step'))]


def get_storage_expes_aggregated_stats(storage_dir, list_of_y_metrics, x_metric, error_bar_type,
                                       data_pickle_file_name):
    expe_dirs = DirectoryTree.get_all_experiments(storage_dir)

    storage_expes_aggregated_stats = defaultdict(lambda: {k: {'x': {}, 'y': {}} for k in list_of_y_metrics})

    if error_bar_type == 'std':
        error_bar_fct = np.std
    elif error_bar_type == 'sem':
        error_bar_fct = lambda x: 2 * np.std(x) / (len(x) ** 0.5)
    else:
        raise NotImplementedError

    for expe_dir in expe_dirs:
        seed_dirs = DirectoryTree.get_all_seeds(expe_dir)
        seed_level_metric_dict = {k: {'y': [], 'x': []} for k in list_of_y_metrics}
        for seed_dir in seed_dirs:
            seed_recorder = Recorder.init_from_pickle_file(
                filename=str(seed_dir / 'recorders' / data_pickle_file_name))

            for k in seed_level_metric_dict.keys():
                x_data, y_data = seed_recorder.aggregate(x_metric=x_metric, y_metric=k,
                                                         aggregation_same_x='mean', remove_none=True)

                seed_level_metric_dict[k]['x'].append(x_data)
                seed_level_metric_dict[k]['y'].append(y_data)

        for k in seed_level_metric_dict.keys():
            storage_expes_aggregated_stats[expe_dir.name][k]['x'] = robust_seed_aggregate(
                seed_level_metric_dict[k]['x'],
                aggregator='strictly_equal')

            storage_expes_aggregated_stats[expe_dir.name][k]['y'] = {'mean':
                robust_seed_aggregate(
                    seed_level_metric_dict[k]['y'],
                    aggregator=np.mean),
                error_bar_type: robust_seed_aggregate(
                    seed_level_metric_dict[k]['y'],
                    aggregator=error_bar_fct)}

    return storage_expes_aggregated_stats


def plot_curve_group(storage_dirs, fig_name, list_of_y_metrics, x_metric, error_bar_type, data_pickle_file_name):
    list_storage_expes_aggregated_stats = {}
    for storage_dir in storage_dirs:
        storage_expes_aggregated_stats = get_storage_expes_aggregated_stats(storage_dir, list_of_y_metrics,
                                                                            x_metric, error_bar_type,
                                                                            data_pickle_file_name)
        expe_dirs = DirectoryTree.get_all_experiments(storage_dir)

        for expe_dir in expe_dirs:
            fig, ax = create_fig((1, 1),figsize=(15,5))
            expe_data = storage_expes_aggregated_stats[expe_dir.name]
            n_curves = len(list(expe_data.items()))
            # plot_curves(ax,
            #             ys=[np.asarray(expe_data[k]['y']['mean']) for k in expe_data.keys()],
            #             xs=[np.asarray(expe_data[k]['x']) for k in expe_data.keys()],
            #             fill_up=[np.asarray(expe_data[k]['y'][error_bar_type]) for k in expe_data.keys()],
            #             fill_down=[np.asarray(expe_data[k]['y'][error_bar_type]) for k in expe_data.keys()],
            #             labels=[k for k in expe_data.keys()],
            #             xlabel='interaction step',
            #             ylabel=fig_name,
            #             legend_pos=(1., 1.),
            #             legend_loc="upper left",
            #             legend_n_columns=1,
            #             legend_font_size=12,
            #             axis_font_size=14,
            #             tick_font_size=12,
            #             )
            plot_curves(ax,
                        ys=[np.asarray(expe_data[k]['y']['mean']) for k in expe_data.keys()],
                        xs=[np.asarray(expe_data[k]['x']) for k in expe_data.keys()],
                        fill_up=[np.asarray(expe_data[k]['y'][error_bar_type]) for k in expe_data.keys()],
                        fill_down=[np.asarray(expe_data[k]['y'][error_bar_type]) for k in expe_data.keys()],
                        xlabel='interaction step',
                        labels=[k for k in expe_data.keys()],
                        legend_pos=(1., 1.),
                        legend_loc="upper left",
                        legend_n_columns=1,
                        ylabel=fig_name,
                        add_legend=True
                        )

            with open('{}/{}.pk'.format(expe_dir,fig_name), 'wb') as fp:
                pickle.dump(expe_data, fp)
            plt.tight_layout()
            fig.savefig(str(expe_dir / f'{fig_name}.png'))
            plt.close(fig)

        list_storage_expes_aggregated_stats[storage_dir.name] = storage_expes_aggregated_stats

    return list_storage_expes_aggregated_stats


if __name__ == "__main__":
    args = get_make_plots_args()

    storage_dirs = select_storage_dirs(args.from_file, args.storage_name, args.root_dir)

    curve_groups_aggregated_results = {}
    for curve_group in args.curve_group_list:
        curve_group_name, metrics = curve_group
        y_metrics_list, x_metric = metrics
        current_curve_group_aggregated_results = plot_curve_group(storage_dirs=storage_dirs,
                                                                  fig_name=curve_group_name,
                                                                  list_of_y_metrics=y_metrics_list,
                                                                  x_metric=x_metric,
                                                                  error_bar_type=args.error_bar_type,
                                                                  data_pickle_file_name=args.data_pickle_file_name)

        curve_groups_aggregated_results[curve_group_name] = current_curve_group_aggregated_results
