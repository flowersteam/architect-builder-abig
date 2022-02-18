import argparse
import pickle

import numpy as np
import random

from alfred.utils.config import parse_bool

from env_comem.gym_gridworld.gridworld import Gridworld

from main_comem.value_iter.algos import SoftValueIteration
from main_comem.agents.agent_policy import TabularSoftMaxPolicy
from main_comem.utils.data_structures import VariableLenBuffer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_size', type=int, nargs='+', default=(6, 5))
    parser.add_argument('--reward_type', type=str, default='sparse',
                        choices=['manhattan', 'sparse', 'progress'])
    parser.add_argument('--obs_type', type=str, default='xy_discrete')
    parser.add_argument('--seed', type=int, default=131214)
    parser.add_argument('--discount_factor', type=float, default=0.95)
    parser.add_argument('--verbose', type=parse_bool, default=True)
    parser.add_argument('--n_episodes', type=int, default=3)

    return parser.parse_args()


def _sanity_test(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    gw = Gridworld(grid_size=args.grid_size,
                   reward_type=args.reward_type,
                   obs_type=args.obs_type,
                   verbose=args.verbose,
                   seed=args.seed)

    value_iter_algo = SoftValueIteration(gw.nS, gw.nA, args.discount_factor)
    policy = TabularSoftMaxPolicy(gw.observation_space, gw.action_space, args.seed)

    gw.seed(args.seed)
    for ep in range(args.n_episodes):
        state, _ = gw.reset()
        Q_values = value_iter_algo.learn(gw.Pmat, gw.Rmat)
        policy.update_params(np.reshape(Q_values, policy.table_dims, order='F'))

        gw.render_goal()
        gw.render()
        i = 0
        ret = 0
        while True:
            action = policy.act(state)
            next_state, reward, done, info = gw.step(action)


            gw.render()
            if gw._on_goal(next_state):
                print(i)
                break
            i += 1
            state = next_state


if __name__ == '__main__':
    args = get_args()
    _sanity_test(args)
