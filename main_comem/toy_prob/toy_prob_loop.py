import numpy as np
from alfred.utils.plots import bar_chart, create_fig
import matplotlib.pyplot as plt
from gym.utils import seeding
from scipy.special import softmax
from gym.spaces import MultiDiscrete, Discrete
import pickle
from main_comem.il_irl.bc import PytorchGradientDescentTabularBC
from main_comem.toy_prob.toy_prob import random_archi_policy, optim_archi_policy, DIFFICULT_POLICY, VARIED_POLICY, \
    EASY_POLICY, UNIFORM_POLICY

if __name__ == '__main__':
    builder_policies = [DIFFICULT_POLICY, EASY_POLICY, VARIED_POLICY, UNIFORM_POLICY]

    update_type = 'proportion'  # 'softmax'
    temperature = .5
    type_archi = 'random'
    if type_archi == 'random':
        archi_policy = random_archi_policy
    else:
        archi_policy = optim_archi_policy

    N_exp = 100
    for builder_policy in builder_policies:
        all_actions = []
        for n in range(N_exp):
            seed = int(np.random.random() * 1e6)
            bc = PytorchGradientDescentTabularBC(model_args={'obs_space': MultiDiscrete([2]), 'act_space': Discrete(2),
                                                             'lr': 0.1,
                                                             'max_epoch': 1000,
                                                             'batch_size': 50,
                                                             'temperature': temperature,
                                                             'seed': seed},
                                                 seed=seed)
            bc.policy.params = builder_policy

            n_iteractions = 40
            n_steps_per_interaction = 100

            np_random, _ = seeding.np_random(seed)

            for iteraction in range(n_iteractions):
                print(f'------------ INTERACTION {iteraction}\n')
                guiding_data = []
                for step in range(n_steps_per_interaction):
                    m = archi_policy(builder_policy=bc.policy.params, np_random=np_random)
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
                accuracy, _, _ = bc.fit(obs, target_a)
                if accuracy.item() == 1.0:
                    break

            final_logits = bc.policy.params
            action = np.zeros(len(final_logits))
            for m in range(len(final_logits)):
                if softmax(final_logits[m])[0] == 0.5 and softmax(final_logits[m])[1] == 0.5:
                    action[m] = -1
                    print('aaa')
                else:
                    action[m] = np.argmax(softmax(final_logits[m]))
            all_actions.append(action)

        histogram = np.zeros((2, 3))

        for i in range(2):
            for j in range(2):
                success = np.sum([1 in a for a in all_actions]) / len(all_actions)
                histogram[i, j] = np.sum([a[i] == j for a in all_actions]) / len(all_actions)
            histogram[i, 2] = np.sum([a[i] == -1 for a in all_actions]) / len(all_actions)
        if (builder_policy == DIFFICULT_POLICY).all():
            out_str = 'difficult'
        elif (builder_policy == EASY_POLICY).all():
            out_str = 'easy'
        elif (builder_policy == UNIFORM_POLICY).all():
            out_str = 'uniform'
        else:
            out_str = 'varied'

        with open('outputs/{}_{}.pk'.format(out_str, type_archi), 'wb') as fp:
            pickle.dump(histogram, fp)
        with open('outputs/success_{}_{}.txt'.format(out_str, type_archi), 'w') as f:
            f.write(str(success))
