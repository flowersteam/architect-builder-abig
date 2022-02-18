import numpy as np
import readchar
import pickle

from env_comem.gym_gridworld.buildworld import Buildworld
from env_comem.com_channel.channel import Channel

from main_comem.mcts.mcts import MCTS
from main_comem.mcts.utils import EnvTransitionFct, EnvRewardFct, PossibleActions
from main_comem.mcts.tree_policy import UCT
from main_comem.mcts.default_policy import HeuristicDefaultPolicy
import main_comem.mcts.utils as mctsutils
from main_comem.utils.data_structures import VariableLenBuffer
from tqdm import tqdm

if __name__ == '__main__':

    data_type = 'goal_conditioning_messages'
    chnl_type = 'identity'
    obs_type = 'tile'
    n_episodes = 1800
    steps_per_episode = 60
    verbose = False
    goal = 'horizontal_line'
    dict_size = 6

    if data_type == 'random_messages_fixed_goal':
        change_goal = False
        messages_type = 'random'
    elif data_type == 'goal_conditioning_messages':
        change_goal = True
        messages_type = 'goal_conditioned'
        goal_message_dict = {'horizontal_line': 0, 'vertical_line': 1, 'grasp_object': 2}
    elif data_type == 'action_conditioning_messages_fixed_goal':
        change_goal = False
        messages_type = 'action_conditioned'
    else:
        raise NotImplementedError

    discount_factor = 0.95

    bw = Buildworld(grid_size=(5, 6),
                    change_goal=change_goal,
                    obs_type=obs_type,
                    verbose=verbose,
                    seed=1234,
                    goal=goal)

    policy = MCTS(state_cast_fct=lambda x: x,
                  transition_fct=EnvTransitionFct(bw.transition_fct),
                  reward_fct=EnvRewardFct(bw.reward_fct),
                  is_terminal_fct=lambda x: False,
                  possible_actions_fct=PossibleActions(bw.action_space, 1234),
                  budget=50,
                  tree_policy=UCT(2 ** 0.5, 12324),
                  default_policy=HeuristicDefaultPolicy(bw.heuristic_value, discount_factor),
                  discount_factor=0.95,
                  keep_tree=True,
                  get_new_root=mctsutils.get_new_root,
                  max_depth=500)

    chnl = Channel(dict_size=dict_size,
                   type=chnl_type,
                   seed=131214)

    buffer = VariableLenBuffer(('gw_obs', 'chnl_obs', 'action'))

    bw.render_legend()

    pbar = tqdm(total=n_episodes)

    for episode in range(n_episodes):
        obs, _ = bw.reset()
        while 'place_object' in bw.goal:
            obs, _ = bw.reset()
        bw.render()
        bw.render()
        for step in range(steps_per_episode):
            bw.render()

            action = policy.act(obs)

            if messages_type == 'random':
                chnl.send(chnl.send_space.sample())
                chnl_obs = chnl.read()
            elif messages_type == 'goal_conditioned':
                chnl.send(goal_message_dict[bw.goal])
                chnl_obs = chnl.read()
            elif messages_type == 'action_conditioned':
                chnl.send(action % dict_size)
                chnl_obs = chnl.read()
            else:
                raise NotImplementedError

            chnl.render(verbose)
            next_obs, reward, _, _ = bw.step(action)

            buffer.append(gw_obs=obs, chnl_obs=chnl_obs, action=action)

            obs = next_obs
        pbar.update()

    with open(f'./buildworld_{goal}_{obs_type}_{chnl_type}_{messages_type}_dict_{dict_size}_nep_{n_episodes}', 'wb') as fh:
        pickle.dump(buffer, fh)
