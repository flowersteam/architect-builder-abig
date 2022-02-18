import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import pickle
from collections import defaultdict, OrderedDict

from alfred.utils.directory_tree import DirectoryTree
from alfred.utils.misc import select_storage_dirs, robust_seed_aggregate
from alfred.utils.plots import create_fig, plot_curves
from alfred.utils.config import load_dict_from_json
from alfred.utils.recorder import Recorder


def read_file(exp_dir, exp_number):
    with open('{}/{}/entropies.pk'.format(exp_dir, exp_number), 'rb') as fp:
        entropies = pickle.load(fp)
    with open('{}/{}/scores.pk'.format(exp_dir, exp_number), 'rb') as fp:
        scores = pickle.load(fp)
    with open('{}/{}/MI.pk'.format(exp_dir, exp_number), 'rb') as fp:
        MI = pickle.load(fp)

    return entropies, scores, MI


def remove_key(dict, key_list):
    out_dict = {}
    for k, v in dict.items():
        if k not in key_list:
            out_dict[k] = v
    return out_dict


def create_comparison_dict(exp_dirs, exp_numbers, exp_names):
    out_dict = {}
    for exp_name, exp_dir, exp_number in zip(exp_names, exp_dirs, exp_numbers):
        out_dict[exp_name] = {}
        entropies, scores, MI = read_file(exp_dir, exp_number)
        out_dict[exp_name].update(remove_key(entropies, ['builder_preferred_entropy']))
        out_dict[exp_name].update(remove_key(scores,
                                             ['architect_accuracy', 'builder_accuracy', 'builder_accuracy_to_init',
                                              'builder_accuracy_to_previous']))
        out_dict[exp_name].update(remove_key(MI,
                                             ['builder_Isma_p', 'builder_Isma_pa',
                                              'builder_Isa_pa', 'builder_Ima_pa']))

    return out_dict


def select_in_dict(big_dict, keys):
    out_dict = OrderedDict()
    for k in big_dict.keys():
        for kk in keys:
            out_dict[k + '_' + kk] = big_dict[k][kk]
    return out_dict


if __name__ == '__main__':
    expe_dir_ABIG = 'your path to ABIG results'
    expe_dir_ABIG_no_intent = 'your path to ABIG-no-intent'
    error_bar_type = 'std'

    exp_dirs = [expe_dir_ABIG, expe_dir_ABIG_no_intent]
    exp_numbers = ['experiment14', 'experiment4']
    exp_names = ['ABIG', 'ABIG-no-intent']

    big_dict = create_comparison_dict(exp_dirs, exp_numbers, exp_names)
    f_size = (10,5)
    fig, ax = create_fig((1, 1),figsize=f_size)
    data = select_in_dict(big_dict, ['builder_policy_entropy'])
    labels = ['ABIG', 'ABIG-no-intent']
    colors = ['#5196C6', '#FB7F11']
    plot_curves(ax,
                ys=[np.asarray(data[k]['y']['mean']) for k in data.keys()],
                xs=[np.asarray(data[k]['x']) for k in data.keys()],
                fill_up=[np.asarray(data[k]['y'][error_bar_type]) for k in data.keys()],
                fill_down=[np.asarray(data[k]['y'][error_bar_type]) for k in data.keys()],
                linewidth=2,
                xlabel='interaction step',
                labels=labels,
                legend_pos=(0.5, 1),
                markers=['x', '+'],
                n_y_ticks=5,
                ylim=[-0.05, 1.95],
                colors=colors,
                markevery=10,
                legend_n_columns=1,
                ylabel='Builder Entropy',
                add_legend=True
                )
    plt.tight_layout()
    plt.savefig('learning_entropy.pdf')

    fig, ax = create_fig((1, 1),figsize=f_size)
    data = select_in_dict(big_dict, ['success'])
    labels = ['ABIG', 'ABIG-no-intent']
    colors = ['#5196C6', '#FB7F11']
    plot_curves(ax,
                ys=[np.asarray(data[k]['y']['mean']) for k in data.keys()],
                xs=[np.asarray(data[k]['x']) for k in data.keys()],
                fill_up=[np.asarray(data[k]['y'][error_bar_type]) for k in data.keys()],
                fill_down=[np.asarray(data[k]['y'][error_bar_type]) for k in data.keys()],
                linewidth=2,
                xlabel='interaction step',
                labels=labels,
                colors=colors,
                legend_pos=(0.2, 1),
                n_y_ticks=5,
                ylim=[-0.05, 1.15],
                markers=['x', '+'],
                markevery=10,
                legend_n_columns=1,
                ylabel='Success Rate',
                add_legend=True
                )
    plt.tight_layout()
    plt.savefig('learning_success.pdf')

    fig, ax = create_fig((1, 1),figsize=f_size)
    data = select_in_dict(big_dict, ['builder_Isa_p', 'builder_Ima_p'])
    labels = ['ABIG $I_s$', 'ABIG $I_m$','ABIG-no-intent $I_s$','ABIG-no-itent $I_m$']
    colors = ['#5196C6', '#124466', '#FB7F11', '#BE600E']
    plot_curves(ax,
                ys=[np.asarray(data[k]['y']['mean']) for k in data.keys()],
                xs=[np.asarray(data[k]['x']) for k in data.keys()],
                fill_up=[np.asarray(data[k]['y'][error_bar_type]) for k in data.keys()],
                fill_down=[np.asarray(data[k]['y'][error_bar_type]) for k in data.keys()],
                linewidth=2,
                xlabel='interaction step',
                labels=labels,
                colors=colors,
                legend_pos=(0.35, 1),
                n_y_ticks=5,
                ylim=[-0.05, 1.15],
                markers=['x', 'X','+','P'],
                markevery=10,
                legend_n_columns=2,
                ylabel='Mutual Information',
                add_legend=True
                )
    plt.tight_layout()
    plt.savefig('learning_MI.pdf')

    stop = 0
