import pickle
import numpy as np
import matplotlib.pyplot as plt
from alfred.utils.plots import bar_chart, create_fig
from main_comem.toy_prob.toy_prob import DIFFICULT_POLICY, EASY_POLICY, VARIED_POLICY, UNIFORM_POLICY


def plot_histogram(dict_histo, dict_init, save_name, alg_name):

    group_names = [f'$m_{i + 1}$' for i in range(len(histogram))]
    n_groups = len(group_names)
    fig, ax = create_fig((1, len(dict_histo.keys())), figsize=(15, 5))

    # scores = {group_name: {f'$a_{i+1}$': prob for i, prob in enumerate(message_probs)}
    #                   for message_probs, group_name in zip(policy, group_names)}

    action_names = {f'$a_1$': '$a_1$ (loose)', '$a_2$': 'draw ($P(a_1|m) = P(a_2|m)$)', '$a_3$': '$a_2$ (win)'}
    if alg_name =='ABIM':
        colors = {list(action_names.values())[0]:'#124466', list(action_names.values())[1]:'#5196C6', list(action_names.values())[2]:'#B7D5E9'}
    else:
        colors = {list(action_names.values())[0]:'#BE600E', list(action_names.values())[1]:'#FB7F11', list(action_names.values())[2]:'#F4C083'}
    for idx, (k,v) in enumerate(dict_histo.items()):
        n_bars_per_group = len(v[0])  # number of actions
        group_names = [f'$m_{i + 1}$' for i in range(len(v))]
        v[:, [1, 2]] = v[:, [2, 1]]
        scores = {
            action_names[f'$a_{i + 1}$']: {group_name: v[j, i] for j, group_name in enumerate(group_names)}
            for i in range(n_bars_per_group)}


        fig, ax = create_fig((1,1), figsize=(5, 5))
        bar_chart(ax,
              scores=scores,
              group_names=group_names,
              xlabel=k,
              ylabel='' if idx==0 else '',
              title="",
              fontsize=15,
              colors=colors,
              fontratio=1.2,
              legend_pos=(0.5, 5),
              ecolor='w',
              ylim=[0., 1.15],
              make_legend=False,
              y_ticks=[0., 1.],
              aspect_ratio=9
              )

        plt.tight_layout()
        fig.savefig('{}_{}.png'.format(save_name, k), bbox_inches='tight')
    # ax[1].set_title('Initial Conditions',fontsize=16)
    # ax[1].set_title('Frequency of final preferred action for each message over 100 seeds',fontsize=16)

    # plt.tight_layout()

    # fig.savefig(f'{save_name}.png', bbox_inches='tight')


if __name__ == '__main__':

    alg_name = 'ABIM-no-intent'
    if alg_name == 'ABIM':
        extra_str = '_optim'
    else:
        extra_str = '_random'
    dict_init = {'difficult':DIFFICULT_POLICY, 'easy':EASY_POLICY, 'varied':VARIED_POLICY}#, 'uniform':UNIFORM_POLICY}
    dict_histo = {}
    for init in ['difficult', 'easy', 'varied']:#,'uniform']:
        with open('outputs/{}{}.pk'.format(init,extra_str), 'rb') as fp:
            histogram = pickle.load(fp)
        dict_histo[init]=histogram

    plot_histogram(dict_histo,dict_init,'outputs/toy_prob_freq_{}'.format(alg_name),alg_name)

