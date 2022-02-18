import pickle
from pathlib import Path
import torch

from main_comem.agents.builder import Builder
from main_comem.agents.architect import HardcodedArchitectPolicy
from env_comem.gym_gridworld.gridworld import Gridworld
from env_comem.gym_gridworld.buildworld import Buildworld
from env_comem.com_channel.channel import Channel
from env_comem.blender import ObsBlender
from main_comem.mcts.mcts import MCTS
from main_comem.mcts.utils import EnvTransitionFct, EnvRewardFct, PossibleActions
from main_comem.mcts.tree_policy import UCT
from main_comem.mcts.default_policy import HeuristicDefaultPolicy
from main_comem.utils.ml import obs_to_torch
from main_comem.utils.ml import to_torch
import main_comem.mcts.utils as mctsutils

if __name__ == "__main__":

    # torch has a centralized seed so we have to set it here
    torch.manual_seed(131214)
    torch.backends.cudnn.deterministic = True

    n_episodes = 100
    steps_per_episode = 30
    messages_type = 'random'

    verbose = True
    goal_message_dict = {'horizontal_line': 0, 'vertical_line': 1, 'grasp_object': 2}


    discount_factor = 0.95
    dict_size = 6
    data_path = 'buildworld_grasp_object_xy_continuous_identity_random_dict_6_nep_600'

    mode = 'test'
    arch = 'mlp_attention'

    if arch == 'film':
        data_path = 'buildworld_horizontal_line_tile_identity_random_dict_6_nep_600'
        blender_obs_type = 'obs_dict'
        to_torch_fct = obs_to_torch
        obs_type = 'tile'
        chnl_type = 'identity'
    elif arch == 'mlp_emb':
        data_path = 'buildworld_grasp_object_xy_continuous_identity_random_dict_6_nep_600'
        blender_obs_type = 'obs_dict_flat_gw'
        to_torch_fct = obs_to_torch
        obs_type = 'xy_continuous'
        chnl_type = 'identity'
    elif arch == 'mlp_attention':
        data_path = 'buildworld_grasp_object_xy_continuous_identity_random_dict_6_nep_600'
        blender_obs_type = 'obs_dict_flat_gw'
        to_torch_fct = obs_to_torch
        obs_type = 'xy_continuous'
        chnl_type = 'identity'
    elif arch == 'mlp':
        obs_type = 'xy_continuous'
        chnl_type = 'one-hot'
        blender_obs_type = None
    else:
        blender_obs_type = None
        to_torch_fct = to_torch

    # gw = Buildworld(grid_size=(5, 6),
    #                reward_type='manhattan',
    #                obs_type='xy_continuous',
    #                verbose=True,
    #                seed=131214)

    bw = Buildworld(grid_size=(5, 6),
                    obs_type=obs_type,
                    change_goal=False,
                    verbose=verbose,
                    seed=131214)

    chnl = Channel(dict_size=dict_size,
                   type=chnl_type,
                   seed=131214)

    obs_blender = ObsBlender(gw_obs_space=bw.observation_space,
                             channel_obs_space=chnl.read_space,
                             type=blender_obs_type)

    builder = Builder(policy_type="bc",
                      policy_args={'budget': 200,
                                   'horizon': 10,
                                   'keep_tree': True,
                                   'max_depth': 5000000,
                                   'ucb_cp': 2 ** 0.5,
                                   'discount_factor': 0.95},
                      obs_blender=obs_blender,
                      gridworld_model=bw,
                      tilde_reward_type="bc_{}_irl".format(arch),
                      tilde_reward_model_args={'obs_space': obs_blender.obs_space,
                                               'act_space': bw.action_space,
                                               'lr': 1e-3,
                                               'max_epoch': 1000,
                                               'batch_size': 256,
                                               'reset_optimizer': True,
                                               'reset_network': False,
                                               'max_wait': 300},
                      seed=131214)

    if messages_type == 'action_conditioned':
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

    if mode == 'train':
        with open(str(Path(__file__).parent / data_path), 'rb') as fh:
            loaded_buffer = pickle.load(fh)

        # overwrite the builder's buffer with loaded data
        builder.buffer = loaded_buffer

        accuracy = builder.update_irl()
        print(f'BC model accuracy: {accuracy}')

        builder.save(f'./builder_{data_path}.pkl')

    if mode == 'test':
        builder = builder.init_from_saved(f'./builder_{data_path}.pkl', obs_blender, bw)

        for episode in range(n_episodes):
            obs, _ = bw.reset()
            while 'place_object' in bw.goal:
                obs, _ = bw.reset()
            bw.render()
            bw.render()
            for step in range(steps_per_episode):
                bw.render()
                if messages_type == 'random':
                    chnl.send(chnl.send_space.sample())
                    chnl_obs = chnl.read()
                elif messages_type == 'goal_conditioned':
                    chnl.send(goal_message_dict[bw.goal])
                    chnl_obs = chnl.read()
                elif messages_type == 'action_conditioned':
                    chnl.send(policy.act(obs) % dict_size)
                    chnl_obs = chnl.read()
                else:
                    raise NotImplementedError

                chnl.render(verbose)
                # action = torch.argmax(
                #     builder.policy(to_torch_fct(obs_blender.blend(obs, chnl_obs)))).numpy()
                action = builder.act(obs_blender.blend(obs, chnl_obs))
                next_obs, reward, _, _ = bw.step(action)

                obs = next_obs
