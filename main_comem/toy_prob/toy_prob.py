import numpy as np
from alfred.utils.plots import bar_chart, create_fig
import matplotlib.pyplot as plt
from gym.utils import seeding
from scipy.special import softmax
from main_comem.il_irl.bc import PytorchGradientDescentTabularBC
from gym.spaces import MultiDiscrete, Discrete

# policies are pol[m,a]

# p(a1\m1) = 0.1, p(a2\m1) = 0.9
# p(a1\m2) = 0.9, p(a2\m2) = 0.1
VARIED_POLICY = np.array([[0.1, 0.9], [0.9, 0.1]])

UNIFORM_POLICY = np.array([[0.5, 0.5], [0.5, 0.5]])

# p(a1\m1) = 0.1, p(a2\m1) = 0.9
# p(a1\m2) = 0.1, p(a2\m2) = 0.9
EASY_POLICY = np.array([[0.2, 0.8], [0.1, 0.9]])

# p(a1\m1) = 0.9, p(a2\m1) = 0.1
# p(a1\m2) = 0.8, p(a2\m2) = 0.2
DIFFICULT_POLICY = np.array([[0.8, 0.2], [0.9, 0.1]])


def render_policy(policy, save_name, first=True):
    n_bars_per_group = len(policy[0])  # number of actions
    group_names = [f'$m_{i + 1}$' for i in range(len(policy))]
    n_groups = len(group_names)
    fig, ax = create_fig((1, 1), figsize=(n_bars_per_group * n_groups, n_groups))

    # scores = {group_name: {f'$a_{i+1}$': prob for i, prob in enumerate(message_probs)}
    #                   for message_probs, group_name in zip(policy, group_names)}
    action_names = {f'$a_1$': '$a_1$ (loose)', '$a_2$': '$a_2$ (win)'}
    scores = {action_names[f'$a_{i + 1}$']: {group_name: policy[j, i] for j, group_name in enumerate(group_names)}
              for i in range(n_bars_per_group)}

    colors = {list(action_names.values())[0]:'#FB7F11', list(action_names.values())[1]:'#F4C083'}
    if first:
        bar_chart(ax,
                  scores=scores,
                  group_names=group_names,
                  ylabel="",
                  fontsize=14.,
                  fontratio=1.2,
                  legend_pos=(0.5,5),
                  colors=colors,
                  ecolor='w',
                  ylim=[0., 1.],
                  make_legend=False,
                  aspect_ratio=7,
                  y_ticks=[0., 1.]
                  )
    else:
        bar_chart(ax,
                  scores=scores,
                  group_names=group_names,
                  fontsize=14.,
                  fontratio=1.2,
                  legend_pos=(0.5, 1.5),
                  colors=colors,
                  ecolor='w',
                  ylim=[0., 1.],
                  make_legend=False,
                  make_y_ticks=False,
                  aspect_ratio=7,
                  y_ticks=[0., 1.]
                  )

    plt.tight_layout()

    fig.savefig(f'{save_name}.png', bbox_inches='tight')


def random_archi_policy(builder_policy, np_random):
    return np_random.randint(0, len(builder_policy[0]))


def optim_archi_policy(builder_policy, np_random):
    # wants to maximize a2
    messages_proba_to_a2 = [message_impact[1] for message_impact in builder_policy]
    return np.argmax(messages_proba_to_a2)


if __name__ == "__main__":
    # seed = 124435
    # builder_policy = DIFFICULT_POLICY
    # archi_policy = optim_archi_policy
    # n_iteractions = 100
    # n_steps_per_interaction = 50
    # temperature = 4.

    # seed = 124435
    # builder_policy = DIFFICULT_POLICY
    # archi_policy = random_archi_policy
    # n_iteractions = 100
    # n_steps_per_interaction = 50
    # temperature = 4.
    # seed = 14
    # builder_policy = DIFFICULT_POLICY
    # archi_policy = optim_archi_policy
    # update_type = 'proportion'  # 'softmax'
    # temperature = .12

    ## JACKPOT
    # seed = 22
    # builder_policy = DIFFICULT_POLICY
    # archi_policy = optim_archi_policy
    # update_type = 'proportion'  # 'softmax'
    # temperature = .5
    # bc = PytorchGradientDescentTabularBC(model_args={'obs_space': MultiDiscrete([2]), 'act_space': Discrete(2),
    #                                                  'lr': 0.1,
    #                                                  'max_epoch': 1000,
    #                                                  'batch_size': 50,
    #                                                  'temperature': temperature,
    #                                                  'seed': seed},
    #                                      seed=seed)

    # seed = 13
    # builder_policy = DIFFICULT_POLICY
    # archi_policy = optim_archi_policy
    # update_type = 'proportion'  # 'softmax'
    # temperature = .5
    # bc = PytorchGradientDescentTabularBC(model_args={'obs_space': MultiDiscrete([2]), 'act_space': Discrete(2),
    #                                                  'lr': 0.1,
    #                                                  'max_epoch': 1000,
    #                                                  'batch_size': 50,
    #                                                  'temperature': temperature,
    #                                                  'seed': seed},
    #                                      seed=seed)

    seed = 13
    builder_policy = DIFFICULT_POLICY
    archi_policy = random_archi_policy
    update_type = 'proportion'  # 'softmax'
    temperature = .5
    bc = PytorchGradientDescentTabularBC(model_args={'obs_space': MultiDiscrete([2]), 'act_space': Discrete(2),
                                                     'lr': 0.1,
                                                     'max_epoch': 1000,
                                                     'batch_size': 50,
                                                     'temperature': temperature,
                                                     'seed': seed},
                                         seed=seed)
    bc.policy.params = builder_policy

    n_iteractions = 10
    n_steps_per_interaction = 100

    np_random, _ = seeding.np_random(seed)
    render_policy(bc.policy.params, 'initial_policy', first=True)

    for iteraction in range(n_iteractions):
        print(f'------------ INTERACTION {iteraction}\n')
        guiding_data = []
        for step in range(n_steps_per_interaction):
            m = archi_policy(builder_policy=bc.policy.params, np_random=np_random)
            # print(f'm={m}')
            # prob = builder_policy[m]
            # print(f'prob={prob}')
            # a = np_random.choice([0, 1], p=prob)
            # print(f'a={a}')
            a = bc.policy.act([m])
            guiding_data.append((m, a))
        obs = []
        target_a = []
        for transition in guiding_data:
            m, a = transition
            obs.append(m)
            target_a.append(a)
        obs = np.array(obs).reshape(-1, 1)
        target_a = np.array(target_a).reshape(-1, 1)
        # reset model
        bc.policy.params = np.zeros(bc.policy.table_dims)
        bc.fit(obs, target_a)
        # builder_policy = np.zeros_like(builder_policy)
        #
        # for transition in guiding_data:
        #     m, a = transition
        #     builder_policy[m, a] += 1
        # if update_type == 'softmax':
        #     for m in range(len(builder_policy)):
        #         builder_policy[m] = softmax(builder_policy[m] / temperature)

        # elif update_type == 'proportion':
        #     for m in range(len(builder_policy)):
        #         sumed = np.sum(builder_policy[m])
        #         if sumed == 0.:
        #             n_a = len(builder_policy[m])
        #             builder_policy[m] = np.asarray([1. for _ in range(n_a)])/n_a
        #         else:
        #             builder_policy[m] = builder_policy[m] / sumed

        to_plot = np.array(bc.policy.params)
        for m in range(len(to_plot)):
            to_plot[m] = softmax(to_plot[m])

        render_policy(to_plot, f'policy_{iteraction}', first=False)
