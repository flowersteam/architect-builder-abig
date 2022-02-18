from env_comem.gym_gridworld.gridworld import Gridworld
from env_comem.gym_gridworld.buildworld import Buildworld
from env_comem.com_channel.channel import Channel
from env_comem.blender import ObsBlender
from main_comem.agents.architect import Architect
from main_comem.agents.builder import Builder

import torch


def make_world(env_type, change_goal, bw_init_goal, n_objects, grid_size, reward_type, obs_type, dict_size,
               com_channel_transformation, obs_blender_type,
               architect_policy_type,
               builder_policy_type, architect_policy_args, builder_policy_args, tilde_builder_type, tilde_builder_args,
               tilde_reward_type, tilde_reward_model_args, seed, verbose):
    # torch has a centralized seed so we have to set it here
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Env
    if env_type == 'gridworld':
        env = Gridworld(n_objects=n_objects,
                        grid_size=grid_size,
                        reward_type=reward_type,
                        obs_type=obs_type,
                        change_goal=change_goal,
                        verbose=verbose,
                        seed=seed)
    else:
        env = Buildworld(n_objects=n_objects,
                         grid_size=grid_size,
                         obs_type=obs_type,
                         change_goal=change_goal,
                         goal=bw_init_goal,
                         verbose=verbose,
                         seed=seed)

    chnl = Channel(dict_size=dict_size,
                   type=com_channel_transformation,
                   seed=seed)

    obs_blender = ObsBlender(gw_obs_space=env.observation_space,
                             channel_obs_space=chnl.read_space,
                             type=obs_blender_type)

    # Agents
    builder = Builder(policy_type=builder_policy_type,
                      policy_args=builder_policy_args,
                      obs_blender=obs_blender,
                      gridworld_model=env,
                      tilde_reward_type=tilde_reward_type,
                      tilde_reward_model_args=tilde_reward_model_args,
                      seed=seed)

    architect = Architect(policy_type=architect_policy_type,
                          policy_args=architect_policy_args,
                          act_space=chnl.send_space,
                          gridworld_model=env,
                          tilde_builder_type=tilde_builder_type,
                          tilde_builder_args=tilde_builder_args,
                          seed=seed,
                          # just for progressive task build-up
                          bypass_world_args={'builder_policy': builder.policy})

    entities = [env, architect, chnl, obs_blender, builder]
    # We don't seed entities here because entities are seeded during construction

    return entities
