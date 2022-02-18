import pickle
from pathlib import Path
import torch

from main_comem.agents.builder import Builder
from main_comem.agents.architect import HardcodedArchitectPolicy
from env_comem.gym_gridworld.gridworld import Gridworld
from env_comem.com_channel.channel import Channel
from env_comem.blender import ObsBlender

if __name__ == "__main__":

    # torch has a centralized seed so we have to set it here
    torch.manual_seed(131214)
    torch.backends.cudnn.deterministic = True

    gw = Gridworld(grid_size=(5, 6),
                   reward_type='manhattan',
                   obs_type='xy_continuous',
                   verbose=True,
                   seed=131214)

    chnl = Channel(dict_size=5,
                   type='one-hot',
                   seed=131214)

    obs_blender = ObsBlender(gw_obs_space=gw.observation_space,
                             channel_obs_space=chnl.read_space)

    builder = Builder(policy_type="bc",
                      policy_args={'budget': 200,
                                   'horizon': 10,
                                   'keep_tree': True,
                                   'max_depth': 5000000,
                                   'ucb_cp': 2 ** 0.5,
                                   'discount_factor': 0.95},
                      obs_blender=obs_blender,
                      gridworld_model=gw,
                      tilde_reward_type="bc_mlp_irl",
                      tilde_reward_model_args={'obs_space': obs_blender.obs_space,
                                               'act_space': gw.action_space,
                                               'lr': 1e-4,
                                               'max_epoch': 1000,
                                               'batch_size': 256},
                      seed=131214)

    architect = HardcodedArchitectPolicy(obs_space=gw.observation_space,
                                         act_space=chnl.send_space,
                                         gw=gw)

    with open(str(Path(__file__).parent / 'task_name_task2_mapping_same_goal_True_300ep.pkl'), 'rb') as fh:
        loaded_buffer = pickle.load(fh)

    # overwrite the builder's buffer with loaded data
    builder.buffer = loaded_buffer

    # we have to transform the message observation because they were recorded for an identity channel
    one_hot_chnl_obs = []
    for message in loaded_buffer._data['chnl_obs']:
        one_hot_chnl_obs.append(chnl._transform(message, 'one-hot'))
    loaded_buffer._data['chnl_obs'] = one_hot_chnl_obs

    loaded_buffer._data['gw_obs'] = loaded_buffer._data['xy_continuous_obs']

    accuracy = builder.update_irl()

    print(f'BC model accuracy: {accuracy}')

    # builder.dump_irl_params(str(Path(__file__).parent / 'irl_policy_params.pkl'))
    #
    # #### To load a pre-trained reward function do
    # ## First create object so that all the object references are correct
    # builder = Builder(policy_type="mcts",
    #                   policy_args={'budget': 200,
    #                                'horizon': 10,
    #                                'keep_tree': True,
    #                                'max_depth': 5000000,
    #                                'ucb_cp': 2 ** 0.5,
    #                                'discount_factor': 0.95},
    #                   obs_blender=obs_blender,
    #                   gridworld_model=gw,
    #                   tilde_reward_type="bc_irl",
    #                   tilde_reward_model_args={'obs_space': obs_blender.obs_space,
    #                                            'act_space': gw.action_space},
    #                   seed=131214)
    #
    # ## Then overwrite the irl_model params
    # builder.load_irl_params(str(Path(__file__).parent / 'irl_policy_params.pkl'))

    # init gridworld
    gw_obs, gw_goal = gw.reset()
    first_goal_gw_internal_state = gw._internal_state
    initial_distance = gw.compute_manhattan_distance(gw_obs)
    gw.render()
    gw.render_goal()
    timestep = 0
    # interaction loop
    while True:

        # architect step
        message = architect.act(gw_obs)

        # communication channel step
        chnl.send(message)
        chnl_obs = chnl.read()
        chnl.render(True)

        blended_obs = obs_blender.blend(gw_obs=gw_obs, chnl_obs=chnl_obs)
        # builder model step

        action = builder.act(blended_obs)

        ## Uncomment following to run bc-policy directly
        # action = builder.irl_algo.policy.act(blended_obs.state)

        # gridworld step
        gw_obs, reward, done, info = gw.step(action)
        gw.render()
        timestep += 1

        if done or timestep >= 20:
            gw_obs, gw_goal = gw.reset()
            first_goal_gw_internal_state['current_state'] = gw._internal_state['current_state']
            gw_obs, gw_goal = gw.reset_to_internal_state(first_goal_gw_internal_state)

            initial_distance = gw.compute_manhattan_distance(gw_obs)
            gw.render()
            gw.render_goal()
            timestep = 0