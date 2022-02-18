import gym
import numpy as np
from scipy.special import softmax
import pickle

from .agent_policy import RandomTabularPolicy, Policy, OptimalGWPolicy, ActionToMessage, TabularSoftMaxPolicy

from main_comem.utils.data_structures import VariableLenBuffer

import main_comem.mcts.utils as mctsutils
from main_comem.mcts.mcts import MCTS
from main_comem.mcts.tree_policy import UCT
from main_comem.mcts.default_policy import MonteCarloReturnPolicy, HeuristicDefaultPolicy
from main_comem.utils.ml import save_checkpoint, load_checkpoint

from main_comem.value_iter.algos import SoftValueIteration

from main_comem.il_irl.bc import TabularBC, DummyTabularBC, BC

from env_comem.com_channel.channel import Channel
from env_comem.blender import ObsBlender


class Architect(object):
    def __init__(self, policy_type, policy_args, act_space, gridworld_model, tilde_builder_type, tilde_builder_args,
                 seed, bypass_world_args=None):
        self.policy_args = policy_args
        self.gridworld_model = gridworld_model
        self.tilde_builder_type = tilde_builder_type
        self.buffer = VariableLenBuffer(('gw_obs', 'message', 'action'))
        self.act_space = act_space
        self.tilde_builder = None
        self.policy_type = policy_type
        self.seed_value = seed

        self._init_dict = {'policy_type': policy_type,
                           'policy_args': policy_args,
                           'act_space': act_space,
                           'gridworld_model': gridworld_model,
                           'tilde_builder_type': tilde_builder_type,
                           'tilde_builder_args': tilde_builder_args,
                           'seed': seed,
                           'bypass_world_args': bypass_world_args}

        ## Builder model definition

        if tilde_builder_type == 'oracle':
            self.tilde_builder = DummyTabularBC()

            self.tilde_builder.set_policy(bypass_world_args['builder_policy'])

        elif tilde_builder_type == 'bc':
            assert isinstance(gridworld_model.observation_space, gym.spaces.MultiDiscrete) \
                   and isinstance(act_space, gym.spaces.Discrete)

            # architect will concatenate messages and obs!
            self.transform_chnl = Channel(dict_size=act_space.n, type='identity', seed=seed)
            self.obs_blender = ObsBlender(gw_obs_space=gridworld_model.observation_space,
                                          channel_obs_space=self.transform_chnl.send_space)

            tilde_builder_obs_space = gym.spaces.MultiDiscrete(list(gridworld_model.observation_space.nvec)
                                                               + [act_space.n])
            self.tilde_builder = TabularBC(model_args={'obs_space': tilde_builder_obs_space,
                                                       'act_space': gridworld_model.action_space},
                                           seed=seed)

        elif tilde_builder_type in ['bc_mlp', 'bc_cnn', 'bc_film', 'bc_mlp_emb', 'bc_mlp_attention']:
            assert isinstance(gridworld_model.observation_space, gym.spaces.Box)
            assert isinstance(act_space, gym.spaces.Discrete)

            if tilde_builder_type == 'bc_mlp':
                # architect will cast messages to one-hots and concatenate to observations (this choice is independent of
                # what the builder is actually doing)!
                self.transform_chnl = Channel(dict_size=act_space.n, type='one-hot', seed=seed)
                network_type = 'mlp'
                obs_blender_type = None

            elif tilde_builder_type == 'bc_cnn':
                # architec will cast message to a new channel in th CNN input
                self.transform_chnl = Channel(dict_size=act_space.n, type='identity', seed=seed)
                network_type = 'cnn'
                obs_blender_type = None

            elif tilde_builder_type == 'bc_film':
                self.transform_chnl = Channel(dict_size=act_space.n, type='identity', seed=seed)
                network_type = 'film'
                obs_blender_type = 'obs_dict'

            elif tilde_builder_type == 'bc_mlp_emb':
                self.transform_chnl = Channel(dict_size=act_space.n, type='identity', seed=seed)
                network_type = 'mlp_emb'
                obs_blender_type = 'obs_dict_flat_gw'

            elif tilde_builder_type == 'bc_mlp_attention':
                self.transform_chnl = Channel(dict_size=act_space.n, type='identity', seed=seed)
                network_type = 'mlp_attention'
                obs_blender_type = 'obs_dict_flat_gw'

            self.obs_blender = ObsBlender(gw_obs_space=gridworld_model.observation_space,
                                          channel_obs_space=self.transform_chnl.read_space,
                                          type=obs_blender_type)
            tilde_builder_args.update({'obs_space': self.obs_blender.obs_space,
                                       'act_space': gridworld_model.action_space,
                                       'network_type': network_type})
            self.tilde_builder = BC(tilde_builder_args, seed=seed)

        elif tilde_builder_type == "none":
            pass
        else:
            raise NotImplementedError(f"{tilde_builder_type}")

        ## Archi policy definition

        if policy_type == 'random':
            self.policy = RandomTabularPolicy(obs_space=None,
                                              act_space=act_space)

        elif policy_type == 'hardcoded_mapping':
            self.policy = HardcodedArchitectPolicy(obs_space=gridworld_model.observation_space,
                                                   act_space=act_space,
                                                   gw=gridworld_model)

        elif policy_type == 'mcts':

            env_transition_fct = ArchitectEnvTransitionFct(
                gw_transition_fct=self.gridworld_model.transition_fct,
                tilde_builder=self.tilde_builder,
                message_cast_fct=lambda x: self.transform_chnl._transform(x, self.transform_chnl._type),
                blend_obs_fct=lambda gw_obs, chnl_obs: self.obs_blender.blend(gw_obs, chnl_obs))

            mcts_transition_fct = mctsutils.EnvTransitionFct(env_transition_fct=env_transition_fct)

            reward_fct = mctsutils.EnvRewardFct(self.gridworld_model.reward_fct)

            is_terminal_fct = lambda x: False
            # BE CAREFUL: THE TERMINAL FUNCTION SHOULD BE HIDDEN TO THE AGENT
            # IF WE USE MANHATTAN REWARDS WITHOUT ABSORBING STATES THE AGENT SHOULD NOT KNOW WHEN THE EPISODE IS DONE
            # OTHERWISE IT IS GOING TO EXPLOIT IT TO GENERATE LONG EPISODES TO COLLECT MORE REWARDS

            possible_actions_fct = mctsutils.PossibleActions(action_space=self.act_space, seed=self.seed_value)

            tree_policy = UCT(cp=self.policy_args['ucb_cp'], seed=self.seed_value)

            if policy_args['use_heuristic']:
                default_policy = HeuristicDefaultPolicy(self.gridworld_model.heuristic_value,
                                                        self.policy_args['discount_factor'])
            else:
                default_policy = MonteCarloReturnPolicy(
                    env_transition_fct=env_transition_fct,
                    env_reward_fct=self.gridworld_model.reward_fct,
                    env_is_terminal_fct=is_terminal_fct,
                    possible_action_fct=possible_actions_fct,
                    discount_factor=self.policy_args['discount_factor'],
                    horizon=self.policy_args['horizon'])

            self.policy = MCTS(state_cast_fct=lambda x: x,
                               transition_fct=mcts_transition_fct,
                               reward_fct=reward_fct,
                               is_terminal_fct=is_terminal_fct,
                               possible_actions_fct=possible_actions_fct,
                               budget=self.policy_args['budget'],
                               tree_policy=tree_policy,
                               default_policy=default_policy,
                               discount_factor=self.policy_args['discount_factor'],
                               keep_tree=self.policy_args['keep_tree'],
                               get_new_root=mctsutils.get_new_root,
                               max_depth=self.policy_args['max_depth'])

        elif policy_type == 'value_iteration':
            self.policy = TabularSoftMaxPolicy(obs_space=self.gridworld_model.observation_space,
                                               act_space=self.act_space,
                                               seed=self.seed_value)

            self.value_iter_alg = SoftValueIteration(nS=self.gridworld_model.nS, nA=self.gridworld_model.nA,
                                                     discount_factor=self.policy_args['discount_factor'])
        else:
            raise NotImplementedError

    def act(self, *args, **kwargs):
        return self.policy.act(*args, **kwargs)

    def store(self, gw_obs, message, action):
        self.buffer.append(gw_obs=gw_obs, message=message, action=action)

    def update_bc(self):
        if self.tilde_builder is None:
            return None, None, None
        data = self.buffer.get_all_current_as_np()
        gw_obs = data['gw_obs']
        message_obs = [self.transform_chnl._transform(m, self.transform_chnl._type) for m in data['message']]
        obs = self.obs_blender.np_batch_blend_raw(gw_obs, message_obs)
        action = data['action']
        acc = self.tilde_builder.fit(obs=obs, target_actions=action)
        return acc

    def update_policy(self):
        if self.policy_type == 'mcts':
            # the mcts policy is automatically updated since it explicitely calls the current reward fct and bc-model
            pass
        elif self.policy_type == 'value_iteration':
            tilde_mat = self.tilde_builder.policy.params

            # the params of the tilde_mat may not be probas since the tabular-irl-policy params are visit counts and not
            # action probas
            sum_over_actions = np.einsum('xyma->xym', tilde_mat)
            if not (sum_over_actions == 1.).all():
                tilde_mat = softmax(tilde_mat, axis=3)

            # reshape the policy matrix so that states are one-dimensional
            # (our env uses Fortran ordering for the indexes)
            one_dim_state_shape = [tilde_mat.shape[0] * tilde_mat.shape[1]] + list(tilde_mat.shape[2:])
            tilde_mat = tilde_mat.reshape(one_dim_state_shape, order='F')

            Pbmat, Rbmat = self.update_value_iter_matrices(Pmat=self.gridworld_model.Pmat,
                                                           Rmat=self.gridworld_model.Rmat,
                                                           builder_policy_mat=tilde_mat)
            Q_values = self.value_iter_alg.learn(Pbmat, Rbmat)
            self.policy.update_params(np.reshape(Q_values, self.policy.table_dims, order='F'))

        elif self.policy_type == "random":
            pass
        elif self.policy_type == "hardcoded_mapping":
            pass
        else:
            raise NotImplementedError

    def update_value_iter_matrices(self, Pmat, Rmat, builder_policy_mat):
        # Pmat is Pmat(s,a,n), n is for next-state
        assert Pmat.shape == (self.value_iter_alg.nS, self.value_iter_alg.nA, self.value_iter_alg.nS)

        # Rmat is Rmat(s,a,n)
        assert Rmat.shape == (self.value_iter_alg.nS, self.value_iter_alg.nA, self.value_iter_alg.nS)

        # builder policy is pi(s,m,a)
        assert builder_policy_mat.shape == (self.value_iter_alg.nS, self.act_space.n,
                                            self.gridworld_model.action_space.n)

        # Pbmat is Pbamt(s,m,n)
        Pbmat = np.einsum('san,sma->smn', Pmat, builder_policy_mat)

        # Rbmat is Rbmat(s,m,n)
        Rbmat = np.einsum('san,sma->smn', Rmat, builder_policy_mat)

        return Pbmat, Rbmat

    def reset_policy(self):
        if type(self.policy) == MCTS:
            self.policy.reset_tree()

    def save(self, architect_file_path):
        if self.tilde_builder is not None:
            tilde_params = self.tilde_builder.policy.get_params()
        else:
            tilde_params = None
        save_checkpoint({'init_dict': self._init_dict,
                         'tilde_builder_policy_params': tilde_params,
                         'policy_params': self.policy.get_params()}, architect_file_path)

    @classmethod
    def init_from_saved(cls, architect_file_path, gridworld_model=None, change_goal=True, config_change_goal=True):
        saved_architect = load_checkpoint(architect_file_path)

        if gridworld_model is not None:
            if not change_goal and not config_change_goal:
                from env_comem.gym_gridworld.gridworld import Gridworld
                if isinstance(gridworld_model, Gridworld):
                    assert saved_architect['init_dict'][
                               'gridworld_model']._goal_state == gridworld_model._goal_state

            saved_architect['init_dict']['gridworld_model'] = gridworld_model

        architect_obj = cls(**saved_architect['init_dict'])

        if architect_obj.tilde_builder_type == 'bc':
            architect_obj.tilde_builder._reward_table = None
            architect_obj.tilde_builder.policy.params = saved_architect['tilde_builder_policy_params']
        elif architect_obj.tilde_builder_type == 'bc_mlp':
            architect_obj.tilde_builder.policy.mlp.load_state_dict(saved_architect['tilde_builder_policy_params'])
        elif architect_obj.tilde_builder_type == 'bc_cnn':
            architect_obj.tilde_builder.policy.cnn.load_state_dict(saved_architect['tilde_builder_policy_params'])
        elif architect_obj.tilde_builder_type == 'bc_film':
            architect_obj.tilde_builder.policy.film_cnn.load_state_dict(saved_architect['tilde_builder_policy_params'])
        elif architect_obj.tilde_builder_type == 'bc_mlp_emb':
            architect_obj.tilde_builder.policy.mlp_emb.load_state_dict(saved_architect['tilde_builder_policy_params'])
        elif architect_obj.tilde_builder_type == 'bc_mlp_attention':
            architect_obj.tilde_builder.policy.mlp_attention.load_state_dict(
                saved_architect['tilde_builder_policy_params'])
        else:
            raise NotImplementedError
        if architect_obj.policy_type == 'value_iteration':
            architect_obj.policy.params = saved_architect['policy_params']
        elif architect_obj.policy_type == 'mcts':
            pass
        else:
            raise NotImplementedError

        return architect_obj


class ArchitectEnvTransitionFct(object):
    def __init__(self, gw_transition_fct, tilde_builder, message_cast_fct, blend_obs_fct):
        self.gw_transition_fct = gw_transition_fct
        self.tilde_builder = tilde_builder
        self.message_cast_fct = message_cast_fct
        self.blend_obs_fct = blend_obs_fct

    def __call__(self, state, action):
        gw_state = state
        chnl_obs = self.message_cast_fct(action)
        obs = self.blend_obs_fct(gw_state, chnl_obs)
        gw_action = self.tilde_builder.act(obs)
        next_state = self.gw_transition_fct(state, gw_action)

        return next_state


class HardcodedArchitectPolicy(Policy):
    def __init__(self, obs_space, act_space, gw):
        super().__init__(obs_space, act_space)
        self.gw_policy = OptimalGWPolicy(gw.observation_space, gw.action_space, gw)
        self.com_protocol = ActionToMessage()

    def act(self, obs):
        action = self.gw_policy.act(obs)
        return self.com_protocol.map(action)
