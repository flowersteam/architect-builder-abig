import argparse
import logging
import numpy as np
import matplotlib
from copy import deepcopy

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from alfred.utils.misc import create_logger, select_storage_dirs
from alfred.utils.config import load_dict_from_json, parse_bool, convert_to_type_from_str, load_config_from_json, \
    validate_config_unique
from alfred.utils.recorder import Recorder
from alfred.utils.plots import plot_curves
from alfred.utils.directory_tree import DirectoryTree

GROUPS_PLOTS_TO_MAKE = [('accuracies_and_success', [('interaction_step', 'success', (None, None), (None, None)),
                                                    ('interaction_step', 'architect_accuracy', (None, None),
                                                     (None, None)),
                                                    (
                                                    'interaction_step', 'builder_accuracy', (None, None), (None, None)),
                                                    ('interaction_step', 'builder_accuracy_to_init', (None, None),
                                                     (None, None)),
                                                    ('interaction_step', 'builder_accuracy_to_previous', (None, None),
                                                     (None, None))]),

                        ('entropies', [('interaction_step', 'builder_preferred_entropy', (None, None), (None, None)),
                                       ('interaction_step', 'builder_policy_entropy', (None, None), (None, None))]),

                        ('mutual_informations', [('interaction_step', 'builder_Isma_p', (None, None), (None, None)),
                                                 ('interaction_step', 'builder_Isa_p', (None, None), (None, None)),
                                                 ('interaction_step', 'builder_Ima_p', (None, None), (None, None)),
                                                 ('interaction_step', 'builder_Isma_pa', (None, None), (None, None)),
                                                 ('interaction_step', 'builder_Isa_pa', (None, None), (None, None)),
                                                 ('interaction_step', 'builder_Ima_pa', (None, None), (None, None))
                                                 ])

                        ]


#
#    (('interaction_step', 'architect_preferred_entropy'), (None, None), (None, None)),
# (('interaction_step', 'architect_n_wait'), (None, None), (None, None)),
# (('interaction_step', 'architect_epoch'), (None, None), (None, None)),
#
# (('interaction_step', 'builder_n_wait'), (None, None), (None, None)),
# (('interaction_step', 'builder_epoch'), (None, None), (None, None)),
# (('interaction_step', 'success'), (None, None), (None, None)),
# (('interaction_step', 'architect_tilde_policy_entropy'), (None, None), (None, None)),
# (('interaction_step', 'architect_tilde_accuracy_to_init'), (None, None), (None, None)),
# (('interaction_step', 'architect_tilde_accuracy_to_previous'), (None, None), (None, None)),
# (('interaction_step', 'builder_accuracy_to_previous'), (None, None), (None, None)),
#
# (('interaction_step', 'architect_Isma_p'), (None, None), (None, None)),
# (('interaction_step', 'architect_Isa_p'), (None, None), (None, None)),
# (('interaction_step', 'architect_Ima_p'), (None, None), (None, None)),
# (('interaction_step', 'builder_Isma_pa'), (None, None), (None, None)),
# (('interaction_step', 'builder_Isa_pa'), (None, None), (None, None)),
# (('interaction_step', 'builder_Ima_pa'), (None, None), (None, None)),
# (('interaction_step', 'architect_Isma_pa'), (None, None), (None, None)),
# (('interaction_step', 'architect_Isa_pa'), (None, None), (None, None)),
# (('interaction_step', 'architect_Ima_pa'), (None, None), (None, None))]


def get_make_plots_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_file', type=str, default=None,
                        help="Path containing all the storage_names for which to make plots")
    parser.add_argument('--storage_name', type=str, default=None)
    parser.add_argument('--groups_plots_to_make', type=plot_definition_parser, nargs='+', default=GROUPS_PLOTS_TO_MAKE,
                        help="To specify the plots: 'x_metric, y_metric, x_min, x_max, y_min, y_max'.\n"
                             "E.g: --plots_to_make \"episode, eval_return, None, None, None, None\" "
                             "\"episode, episode_len, None, None, 0, 350\""
                             "(note: be mindful to put the whole plot definition inside quotes)")

    parser.add_argument('--remove_none', type=parse_bool, default=True,
                        help="The Recorder records None for all keys that are not specified in new_recordings "
                             "(see alfred.utils.recorder.Recorder.write_to_tape()). That allows to associate "
                             "all values of the recorder with each other by the moment they were recorded. "
                             "However, because some values might be recorded much more often than others, some "
                             "metrics will have a lot of None's on their recording-tape, which can make the plots "
                             "look empty. Choosing --remove_nones=True removes all Nones from the tape. However "
                             "for this to work, the choosent --x_metric's and --y_metric's sizes must match.")
    parser.add_argument('--aggregation_same_x', type=str, default='none', choices=['none', 'mean'])
    parser.add_argument('--data_pickle_file_name', type=str, default='train_recorder.pkl')
    parser.add_argument('--one_expe_per_row', type=parse_bool, default=True)
    parser.add_argument('--make_bar_plot', type=parse_bool, default=False)
    parser.add_argument('--root_dir', default=None, type=str)
    return parser.parse_args()


def plot_definition_parser(to_parse):
    def_args = to_parse.split(',')
    x_metric, y_metric, x_min, x_max, y_min, y_max = [convert_to_type_from_str(argument) for argument in def_args]
    pass
    return (x_metric, y_metric, (x_min, x_max), (y_min, y_max))


def create_plot_arrays(from_file, storage_name, root_dir, remove_none, aggregation_same_x,
                       make_bar_plot, one_expe_per_row, logger, groups_plots_to_make=GROUPS_PLOTS_TO_MAKE,
                       data_pickle_file_name='train_recorder.pkl'):
    """
    Creates and and saves comparative figure containing a plot of total reward for each different experiment
    :param storage_dir: pathlib.Path object of the model directory containing the experiments to compare
    :param plots_to_make: list of strings indicating which comparative plots to make
    :return: None
    """
    # Select storage_dirs to run over

    storage_dirs = select_storage_dirs(from_file, storage_name, root_dir)

    for storage_dir in storage_dirs:

        # Get all experiment directories and sorts them numerically

        sorted_experiments = DirectoryTree.get_all_experiments(storage_dir)

        all_seeds_dir = []
        for experiment in sorted_experiments:
            all_seeds_dir = all_seeds_dir + DirectoryTree.get_all_seeds(experiment)

        # Determines what type of search was done

        if (storage_dir / 'GRID_SEARCH').exists():
            search_type = 'grid'
        elif (storage_dir / 'RANDOM_SEARCH').exists():
            search_type = 'random'
        else:
            search_type = 'unknown'

        # Determines row and columns of subplots

        if search_type == 'grid' and not one_expe_per_row:
            variations = load_dict_from_json(filename=str(storage_dir / 'variations.json'))

            # experiment_groups account for the fact that all the experiment_dir in a storage_dir may have been created
            # though several runs of prepare_schedule.py, and therefore, many "groups" of experiments have been created
            experiment_groups = {key: {} for key in variations.keys()}
            for group_key, properties in experiment_groups.items():
                properties['variations'] = variations[group_key]

                properties['variations_lengths'] = {k: len(properties['variations'][k])
                                                    for k in properties['variations'].keys()}

                # Deleting alg_name and task_name from variations (because they will not be contained in same storage_dir)

                hyperparam_variations_lengths = deepcopy(properties['variations_lengths'])
                del hyperparam_variations_lengths['alg_name']
                del hyperparam_variations_lengths['task_name']

                i_max = sorted(hyperparam_variations_lengths.values())[-1]
                j_max = int(np.prod(sorted(hyperparam_variations_lengths.values())[:-1]))

                if i_max < 4 and j_max == 1:
                    # If only one hyperparameter was varied over, we order plots on a line
                    j_max = i_max
                    i_max = 1
                    ax_array_dim = 1

                elif i_max >= 4 and j_max == 1:
                    # ... unless there are 4 or more variations, then we put them in a square-ish fashion
                    j_max = int(np.sqrt(i_max))
                    i_max = int(np.ceil(float(i_max) / float(j_max)))
                    ax_array_dim = 2

                else:
                    ax_array_dim = 2

                properties['ax_array_shape'] = (i_max, j_max)
                properties['ax_array_dim'] = ax_array_dim

        else:
            experiment_groups = {"all": {}}
            for group_key, properties in experiment_groups.items():
                i_max = len(sorted_experiments)  # each experiment is on a different row
                j_max = len(all_seeds_dir) // i_max  # each seed is on a different column

                if i_max == 1:
                    ax_array_dim = 1
                else:
                    ax_array_dim = 2

                properties['ax_array_shape'] = (i_max, j_max)
                properties['ax_array_dim'] = ax_array_dim

        for group_key, properties in experiment_groups.items():
            logger.debug(f"\n===========================\nPLOTS FOR EXPERIMENT GROUP: {group_key}")
            i_max, j_max = properties['ax_array_shape']
            ax_array_dim = properties['ax_array_dim']

            first_exp = group_key.split('-')[0] if group_key != "all" else 0
            if first_exp != 0:
                for seed_idx, seed_dir in enumerate(all_seeds_dir):
                    if seed_dir.parent.stem.strip('experiment') == first_exp:
                        first_seed_idx = seed_idx
                        break
            else:
                first_seed_idx = 0

            for group_plot_to_make in groups_plots_to_make:

                # Creates the subplots
                fig, ax_array = plt.subplots(i_max, j_max, figsize=(10 * j_max, 6 * i_max))

                fig_title, plots_to_make = group_plot_to_make
                for plot_to_make in plots_to_make:

                    x_metric, y_metric, x_lim, y_lim = plot_to_make
                    labels = [y_metric]
                    logger.debug(f'\n{y_metric} as a function of {x_metric}:')

                    for i in range(i_max):

                        for j in range(j_max):
                            if ax_array_dim == 1 and i_max == 1 and j_max == 1:
                                current_ax = ax_array
                            elif ax_array_dim == 1 and (i_max > 1 or j_max > 1):
                                current_ax = ax_array[j]
                            elif ax_array_dim == 2:
                                current_ax = ax_array[i, j]
                            else:
                                raise Exception('ax_array should not have more than two dimensions')

                            try:
                                seed_dir = all_seeds_dir[first_seed_idx + (i * j_max + j)]
                                if group_key != 'all' \
                                        and (
                                        int(str(seed_dir.parent).split('experiment')[1]) < int(group_key.split('-')[0]) \
                                        or int(str(seed_dir.parent).split('experiment')[1]) > int(
                                    group_key.split('-')[1])):
                                    raise IndexError
                                logger.debug(str(seed_dir))
                            except IndexError as e:
                                logger.debug(f'experiment{i * j_max + j} does not exist')
                                current_ax.text(0.2, 0.2, "no experiment\n found",
                                                transform=current_ax.transAxes, fontsize=24, fontweight='bold',
                                                color='red')
                                continue

                            logger.debug(seed_dir)

                            # Writes unique hyperparameters on plot

                            config = load_config_from_json(filename=str(seed_dir / 'config.json'))
                            config_unique_dict = load_dict_from_json(filename=str(seed_dir / 'config_unique.json'))
                            validate_config_unique(config, config_unique_dict)

                            if search_type == 'grid' and not one_expe_per_row:
                                sorted_keys = sorted(config_unique_dict.keys(),
                                                     key=lambda item: (properties['variations_lengths'][item], item),
                                                     reverse=True)

                            else:
                                sorted_keys = config_unique_dict

                            info_str = f'{seed_dir.parent.stem}\n' + '\n'.join(
                                [f'{k} = {config_unique_dict[k]}' for k in sorted_keys])
                            bbox_props = dict(facecolor='gray', alpha=0.1)
                            current_ax.text(0.05, 0.95, info_str, transform=current_ax.transAxes, fontsize=12,
                                            verticalalignment='top', bbox=bbox_props)

                            # Skip cases of UNHATCHED or CRASHED experiments

                            if (seed_dir / 'UNHATCHED').exists():
                                logger.debug('UNHATCHED')
                                current_ax.text(0.2, 0.2, "UNHATCHED",
                                                transform=current_ax.transAxes, fontsize=24, fontweight='bold',
                                                color='blue')
                                continue

                            if (seed_dir / 'CRASH.txt').exists():
                                logger.debug('CRASHED')
                                current_ax.text(0.2, 0.2, "CRASHED",
                                                transform=current_ax.transAxes, fontsize=24, fontweight='bold',
                                                color='red')
                                continue

                            try:

                                # Loading the recorder

                                loaded_recorder = Recorder.init_from_pickle_file(
                                    filename=str(seed_dir / 'recorders' / data_pickle_file_name))

                                # Checking if provided metrics are present in the recorder

                                if y_metric not in loaded_recorder.tape.keys() and y_metric is not None:
                                    logger.debug(f"'{y_metric}' was not recorded in {data_pickle_file_name}")
                                    current_ax.text(0.2, 0.2, "ABSENT METRIC", transform=current_ax.transAxes,
                                                    fontsize=24, fontweight='bold', color='red')
                                    continue

                                if x_metric not in loaded_recorder.tape.keys() and x_metric is not None:
                                    if x_metric is None:
                                        pass
                                    else:
                                        logger.debug(f"'{x_metric}' was not recorded in {data_pickle_file_name}")
                                        current_ax.text(0.2, 0.2, "ABSENT METRIC", transform=current_ax.transAxes,
                                                        fontsize=24, fontweight='bold', color='red')
                                        continue

                                # Removing None entries and aggregating
                                loaded_recorder.tape[x_metric], loaded_recorder.tape[y_metric] = \
                                    loaded_recorder.aggregate(x_metric, y_metric, aggregation_same_x, remove_none)

                                # Plotting

                                try:

                                    if make_bar_plot:
                                        data = loaded_recorder.tape
                                        keys = list(data.keys())

                                        def post_process_group_name(group_name):
                                            str_list = group_name.split(',', 2)
                                            str_list = [st.split(':', 2) for st in str_list]
                                            final_group_name = [st[1] for st in str_list]
                                            final_group_name = ','.join(final_group_name)
                                            return final_group_name

                                        group_names = [post_process_group_name(key) for key in keys]

                                        heights = [data[key] for key in keys]

                                        current_ax.bar(x=range(len(group_names)),
                                                       height=heights,
                                                       tick_label=group_names)
                                    else:
                                        if x_metric is not None:
                                            plot_curves(current_ax,
                                                        ys=[loaded_recorder.tape[y_metric]],
                                                        xs=[loaded_recorder.tape[x_metric]],
                                                        xlim=x_lim,
                                                        ylim=y_lim,
                                                        xlabel=x_metric,
                                                        labels=labels)
                                        else:
                                            plot_curves(current_ax,
                                                        ys=[loaded_recorder.tape[y_metric]],
                                                        xlim=x_lim,
                                                        ylim=y_lim,
                                                        labels=labels)

                                except Exception as e:
                                    logger.debug(f'Polotting error: {e}')

                            except FileNotFoundError:
                                logger.debug('Training recorder not found')
                                current_ax.text(0.2, 0.2, f"'{data_pickle_file_name}'\nnot found",
                                                transform=current_ax.transAxes, fontsize=24, fontweight='bold',
                                                color='red')
                                continue

                plt.tight_layout()
                fig.savefig(str(storage_dir / f'{fig_title}.png'))
                plt.close(fig)


if __name__ == '__main__':
    logger = create_logger("PLOTS", logging.DEBUG, logfile=None)
    kwargs = vars(get_make_plots_args())
    create_plot_arrays(**kwargs, logger=logger)
