from gym import spaces
from gym.utils import seeding
import numpy as np
import pickle
from pathlib import Path

from env_comem.blender import BlendedObs
from .agent_policy import RandomTabularPolicy, StateConditionedMessageToActionBijectiveMapping, \
    StateConditionedMessageToActionInjectiveMapping, StateConditionedMessageToActionStochasticMapping, \
    TabularSoftMaxPolicy
from main_comem.mcts.tree_policy import UCT
from main_comem.mcts.node import DecisionNode
from main_comem.mcts.mcts import MCTS
from main_comem.il_irl.bc import TabularBC, BC
from main_comem.utils.data_structures import VariableLenBuffer
from main_comem.value_iter.algos import FactoredSoftValueIteration
from main_comem.utils.ml import save_checkpoint, load_checkpoint

DEF_CONCENTRATION_PARAM = .2


class Builder(object):
    def __init__(self, policy_type, policy_args, obs_blender, gridworld_model,
                 tilde_reward_type, tilde_reward_model_args, seed):
        self.policy_args = policy_args
        self.obs_blender = obs_blender
        self.gridworld_model = gridworld_model
        self.tilde_reward_type = tilde_reward_type
        self.buffer = VariableLenBuffer(('gw_obs', 'chnl_obs', 'action'))
        self.irl_algo = None
        self.policy_type = policy_type
        self.seed_value = seed
        self._init_dict = {'policy_type': policy_type,
                           'policy_args': policy_args,
                           'obs_blender': obs_blender,
                           'gridworld_model': gridworld_model,
                           'tilde_reward_type': tilde_reward_type,
                           'tilde_reward_model_args': tilde_reward_model_args,
                           'seed': seed}

        #### builder reward model

        if tilde_reward_type == 'oracle':
            rollout_reward_fct = GridworldOnlyReward(env_reward=self.gridworld_model._reward_fct)

        elif tilde_reward_type == 'action_message_mapping':
            rollout_reward_fct = ActionMappingReward()

        elif tilde_reward_type == 'action_message_mapping_with_goal_info':
            rollout_reward_fct = ActionMappingRewardWithGoalInfo(
                distance_fct=self.gridworld_model.compute_manhattan_distance)

        elif tilde_reward_type == 'bc_irl':
            model_args = {'act_space': gridworld_model.action_space, 'obs_space': obs_blender.obs_space}

            self.irl_algo = TabularBC(model_args=model_args, seed=seed)

            # we have to initialize that initial reward_table to something different than 0 everywhere
            # otherwise the mcts is just a uniform policy

            if tilde_reward_model_args['init'] == 'bijective':
                # the reward strictly prefers one action for each message so we change the reward table
                # into its one-hot equivalent

                import numpy as np

                init_policy = StateConditionedMessageToActionBijectiveMapping(obs_space=obs_blender.obs_space,
                                                                              act_space=gridworld_model.action_space)
                one_hots = init_policy.params
                initial_reward_table = one_hots

            elif tilde_reward_model_args['init'] == 'stochastic':
                initial_reward_table = StateConditionedMessageToActionStochasticMapping(
                    obs_space=obs_blender.obs_space,
                    act_space=gridworld_model.action_space,
                    concentration_param=.2).probas
            else:
                raise NotImplementedError

            self.irl_algo._reward_table = initial_reward_table
            rollout_reward_fct = Reward(self.irl_algo.reward_fct)

        elif tilde_reward_type in ['bc_mlp_irl', 'bc_cnn_irl', 'bc_film_irl', 'bc_mlp_emb_irl', 'bc_mlp_attention_irl']:
            if 'mlp' in tilde_reward_type:
                network_type = 'mlp'
                if 'emb' in tilde_reward_type:
                    network_type = 'mlp_emb'
                if 'attention' in tilde_reward_type:
                    network_type = 'mlp_attention'
            elif 'cnn' in tilde_reward_type:
                network_type = 'cnn'
            elif 'film' in tilde_reward_type:
                network_type = 'film'
            else:
                raise NotImplementedError

            tilde_reward_model_args.update(
                {'act_space': gridworld_model.action_space, 'obs_space': obs_blender.obs_space,
                 'network_type': network_type})
            self.irl_algo = BC(model_args=tilde_reward_model_args, seed=seed)
            rollout_reward_fct = lambda x: NotImplementedError

        else:
            raise NotImplementedError

        self.reward_fct = rollout_reward_fct

        #### builder policy

        if policy_type == 'random':
            self.policy = RandomTabularPolicy(obs_space=None,
                                              act_space=gridworld_model.action_space)

        elif policy_type == 'mcts':

            mcts_reward_fct = TwoStageMctsRewardFct(rollout_reward_fct)

            transition_fct = TwoStageMctsTransitionFct(gw_transition_fct=self.gridworld_model._transition_fct,
                                                       obs_blender=self.obs_blender)
            possible_action_fct = TwoStageMctsPossibleActions(
                action_space={'gw_action_space': self.gridworld_model.action_space,
                              'com_channel_action_space': self.obs_blender.channel_obs_space},
                seed=self.seed_value)

            tree_policy = UCT(cp=self.policy_args['ucb_cp'], seed=self.seed_value)

            default_policy = TwoStageMonteCarloReturnPolicy(env_transition_fct=self.gridworld_model._transition_fct,
                                                            obs_blender=self.obs_blender,
                                                            env_reward_fct=rollout_reward_fct,
                                                            env_is_terminal_fct=lambda x: False,
                                                            # BE CAREFUL: THE TERMINAL FUNCTION SHOULD BE HIDDEN TO THE AGENT
                                                            # IF WE USE MANHATTAN REWARDS WITHOUT ABSORBING STATES THE AGENT SHOULD NOT KNOW WHEN THE EPISODE IS DONE
                                                            # OTHERWISE IT IS GOING TO EXPLOIT IT TO GENERATE LONG EPISODES TO COLLECT MORE REWARDS,
                                                            possible_action_fct=possible_action_fct,
                                                            discount_factor=self.policy_args['discount_factor'],
                                                            horizon=self.policy_args['horizon'])

            self.policy = MCTS(state_cast_fct=state_cast_fct,
                               transition_fct=transition_fct,
                               reward_fct=mcts_reward_fct,
                               is_terminal_fct=lambda x: False,
                               possible_actions_fct=possible_action_fct,
                               budget=self.policy_args['budget'],
                               tree_policy=tree_policy,
                               default_policy=default_policy,
                               discount_factor=self.policy_args['discount_factor'],
                               keep_tree=self.policy_args['keep_tree'],
                               get_new_root=get_new_root,
                               max_depth=self.policy_args['max_depth'])

        elif policy_type == 'value_iteration':
            self.policy = TabularSoftMaxPolicy(obs_space=self.obs_blender.obs_space,
                                               act_space=self.gridworld_model.action_space,
                                               seed=self.seed_value)

            self.value_iter_alg = FactoredSoftValueIteration(nS=self.gridworld_model.nS,
                                                             nM=self.obs_blender.channel_obs_space.n,
                                                             nA=self.gridworld_model.nA,
                                                             message_model_type=self.policy_args['message_model_type'],
                                                             discount_factor=self.policy_args['discount_factor'])

        elif policy_type == 'bc':
            assert self.tilde_reward_type in ['bc_irl', 'bc_mlp_irl', 'bc_cnn_irl', 'bc_film_irl', 'bc_mlp_emb_irl',
                                              'bc_mlp_attention_irl']
            self.policy = self.irl_algo.policy

        elif policy_type == 'bijective_mapping':
            self.policy = StateConditionedMessageToActionBijectiveMapping(obs_space=obs_blender.obs_space,
                                                                          act_space=gridworld_model.action_space)

        elif policy_type == 'injective_mapping':
            self.policy = StateConditionedMessageToActionInjectiveMapping(obs_space=obs_blender.obs_space,
                                                                          act_space=gridworld_model.action_space)

        elif policy_type == 'stochastic_mapping':
            self.policy = StateConditionedMessageToActionStochasticMapping(obs_space=obs_blender.obs_space,
                                                                           act_space=gridworld_model.action_space,
                                                                           concentration_param=DEF_CONCENTRATION_PARAM)
        else:
            raise NotImplementedError

    def act(self, *args, **kwargs):
        return self.policy.act(*args, **kwargs)

    def update_irl(self):
        if self.irl_algo is None:
            return
        data = self.buffer.get_all_current_as_np()
        obs = self.obs_blender.np_batch_blend_raw(batch_gw_obs=data['gw_obs'], batch_chnl_obs=data['chnl_obs'])
        action = data['action']
        acc = self.irl_algo.fit(obs=obs, target_actions=action)

        if self.policy_type == "bc":
            # this is needed if the irl_algo defines a new policy (for example with reset_networks)
            self.policy = self.irl_algo.policy
        return acc

    def update_policy(self):
        if self.policy_type == 'mcts':
            # the mcts policy is automatically updated since it explicitely calls the current reward fct and bc-model
            pass

        elif self.policy_type == 'value_iteration':

            Rmat = np.ones((self.value_iter_alg.nS, self.value_iter_alg.nM, self.value_iter_alg.nA,
                            self.value_iter_alg.nS, self.value_iter_alg.nM))

            if self.tilde_reward_type in ['oracle', 'action_message_mapping', 'action_message_mapping_with_goal_info']:
                # calls the reward_fct with the change of state space (tabular to x,y) to build the reward
                # matrix
                Rmat = self.value_iter_alg.build_Rmat_from_reward_fct(self.reward_fct_one_dim_state)

            elif self.tilde_reward_type == 'bc_irl':

                # because the env is deterministic the irl reward table will have shape (nS, nM, nA)
                Rmat_irl = self.irl_algo.reward_table
                one_dim_state_shape = [Rmat_irl.shape[0] * Rmat_irl.shape[1]] + list(Rmat_irl.shape[2:])
                Rmat_irl = Rmat_irl.reshape(one_dim_state_shape, order='F')

                # this makes the broadcast that we want
                Rmat = np.einsum('smank,sma->smank', Rmat, Rmat_irl)

            else:
                raise NotImplementedError

            Q_values = self.value_iter_alg.learn(Pmat_env=self.gridworld_model.Pmat,
                                                 Rmat=Rmat)
            self.policy.update_params(np.reshape(Q_values, self.policy.table_dims, order='F'))
        elif self.policy_type == 'bc':
            ## the bc policy is already linked to the irl algo
            pass
        else:
            raise NotImplementedError

    def reward_fct_one_dim_state(self, s, m, a, sp, mp):
        two_dim_s = self.gridworld_model.one_dim_state_to_two_dim_state(s)
        two_dim_sp = self.gridworld_model.one_dim_state_to_two_dim_state(sp)

        blended_obs = self.obs_blender.blend(two_dim_s, m)
        next_blended_obs = self.obs_blender.blend(two_dim_sp, mp)

        return self.reward_fct(blended_obs, a, next_blended_obs)

    def store(self, gw_obs, chnl_obs, action):
        self.buffer.append(gw_obs=gw_obs, chnl_obs=chnl_obs, action=action)

    def reset_policy(self):
        if type(self.policy) == MCTS:
            self.policy.reset_tree()

    def dump_irl_params(self, file_path):
        with open(file_path, 'wb') as fh:
            pickle.dump(self.irl_algo.policy.params, fh)
            fh.close()

    def load_irl_params(self, file_path):
        with open(file_path, 'rb') as fh:
            self.irl_algo.policy.params = pickle.load(fh)
            fh.close()
        self.irl_algo._reward_table = None

    def save(self, builder_file_path):
        save_checkpoint({'init_dict': self._init_dict,
                         'irl_algo_policy_params': self.irl_algo.policy.get_params(),
                         'policy_params': self.policy.get_params()}, builder_file_path)

    @classmethod
    def init_from_saved(cls, builder_file_path, obs_blender=None, gridworld_model=None):
        saved_builder = load_checkpoint(builder_file_path)

        if gridworld_model is not None:
            saved_builder['init_dict']['gridworld_model'] = gridworld_model

        if obs_blender is not None:
            saved_builder['init_dict']['obs_blender'] = obs_blender

        # # todo: REMOVE
        # saved_builder['init_dict']['tilde_reward_model_args'].update({'reset_network':False})

        builder_obj = cls(**saved_builder['init_dict'])

        if builder_obj.tilde_reward_type == 'bc_irl':
            builder_obj.irl_algo._reward_table = None
            builder_obj.irl_algo.policy.params = saved_builder['irl_algo_policy_params']
        elif builder_obj.tilde_reward_type == 'bc_mlp_irl':
            builder_obj.irl_algo.policy.mlp.load_state_dict(saved_builder['irl_algo_policy_params'])
        elif builder_obj.tilde_reward_type == 'bc_cnn_irl':
            builder_obj.irl_algo.policy.cnn.load_state_dict(saved_builder['irl_algo_policy_params'])
        elif builder_obj.tilde_reward_type == 'bc_film_irl':
            builder_obj.irl_algo.policy.film_cnn.load_state_dict(saved_builder['irl_algo_policy_params'])
        elif builder_obj.tilde_reward_type == 'bc_mlp_emb_irl':
            builder_obj.irl_algo.policy.mlp_emb.load_state_dict(saved_builder['irl_algo_policy_params'])
        elif builder_obj.tilde_reward_type == 'bc_mlp_attention_irl':
            builder_obj.irl_algo.policy.mlp_attention.load_state_dict(saved_builder['irl_algo_policy_params'])
        else:
            raise NotImplementedError

        if builder_obj.policy_type == 'value_iteration':
            builder_obj.policy.params = saved_builder['policy_params']
        elif builder_obj.policy_type == 'bc':
            assert builder_obj.tilde_reward_type in ['bc_irl', 'bc_mlp_irl', 'bc_cnn_irl', 'bc_film_irl',
                                                     'bc_mlp_emb_irl', 'bc_mlp_attention_irl']
            builder_obj.policy = builder_obj.irl_algo.policy
        else:
            raise NotImplementedError

        return builder_obj


def get_new_root(state, root_node, picked_action):
    blended_obs = state[1]

    picked_action_chance_node = root_node.children[picked_action]
    gw_obs = ('gw_obs', blended_obs._obs['gw_obs'])
    chnl_obs = blended_obs._obs['channel_obs']

    if gw_obs in picked_action_chance_node.children:
        sampled_gw_obs_node = picked_action_chance_node.children[gw_obs]
        if chnl_obs in sampled_gw_obs_node.children:
            picked_chnl_obs_chance_node = sampled_gw_obs_node.children[chnl_obs]
            if state in picked_chnl_obs_chance_node.children:
                return picked_chnl_obs_chance_node.children[state]

    return None


def state_cast_fct(state):
    if isinstance(state, BlendedObs):
        return tuple(['blended_obs', state])
    # test if state is iterable as in two-stage mcts state are (str, obj): (state_type, state_value)
    elif hasattr(state, '__iter__'):
        if isinstance(state[0], str):
            return state
        else:
            return tuple(['gw_obs', tuple(state)])
    else:
        return tuple(['gw_obs', tuple(state)])


class TwoStageMctsPossibleActions(object):
    # two stages decision nodes will be either ('gw_obs', state) or ('blended_obs', BlendedObs)

    # from a blendedObs we have to take a gw_action
    # from a gw_obs we have to take a message

    def __init__(self, action_space, seed):
        # action_space should be a dict like
        # {'gw_action_space': action_space, 'com_channel_action_space': action_space}

        assert isinstance(action_space, dict)
        self.action_space = action_space

        # for the moment we only deal with discrete action spaces
        assert all([isinstance(act_space, spaces.Discrete) for act_space in action_space.values()])

        self.state_to_action_space = {'gw_obs': self.action_space['com_channel_action_space'],
                                      'blended_obs': self.action_space['gw_action_space']}

        self.np_random, _ = seeding.np_random(None)
        self.seed(seed)

    def seed(self, seed):
        self.np_random.seed(seed)

    def __call__(self, state):
        corresponding_action_space = self.state_to_action_space[state[0]]  # extract action type from state

        possible_actions = list(range(corresponding_action_space.n))
        self.np_random.shuffle(possible_actions)
        return possible_actions

    def sample(self, action_space_key):
        # this fct will be use by the TwoStageMonteCarloPolicy to randomly select actions
        return self.action_space[action_space_key].sample()


class TwoStageMctsTransitionFct(object):
    def __init__(self, gw_transition_fct, obs_blender):
        self.gw_transition_fct = gw_transition_fct
        self.obs_blender = obs_blender

    def __call__(self, decision_node, action):
        assert isinstance(decision_node, DecisionNode)
        state = decision_node.state
        assert isinstance(state, tuple)
        obs_type = state[0]
        assert isinstance(obs_type, str)

        # if the state is a blended obs the action is a builder action and we must transition in the gw
        if obs_type == 'blended_obs':
            gw_obs = state[1]._obs['gw_obs']
            next_gw_obs = self.gw_transition_fct(state=gw_obs, action=action)
            assert hasattr(next_gw_obs, '__hash__')
            return tuple(['gw_obs', next_gw_obs])

        # if the state is a gw obs, the action is a potential channel obs and we must blend them into a blended_obs
        elif obs_type == 'gw_obs':
            gw_obs = state[1]
            new_blended_obs = self.obs_blender.blend(gw_obs=gw_obs, chnl_obs=action)
            return tuple(['blended_obs', new_blended_obs])

        else:
            raise NotImplementedError


class TwoStageMctsRewardFct(object):
    def __init__(self, reward_fct):
        self.reward_fct = reward_fct

    def __call__(self, decision_node, action, next_state):
        next_obs_type = next_state[0]

        # the builder's reward function only deals with blended obs
        if next_obs_type == 'gw_obs':
            return None

        elif next_obs_type == 'blended_obs':
            # we get the blended_obs
            next_blended_obs = next_state[1]

            # we get the previous blended observation from the last decision node
            current_blended_obs = decision_node.parent.parent.state[1]

            #### Warning !!! If we are in this case it means that the action is actually a message taken by
            #### the MCTS and not an agent's action !
            #### the real action that we want is given by the parent chance node !!!
            agent_action = decision_node.parent.action

            rew = self.reward_fct(blended_obs=current_blended_obs, action=agent_action,
                                  next_blended_obs=next_blended_obs)
            return rew

        else:
            raise NotImplementedError


class TwoStageMonteCarloReturnPolicy(object):
    def __init__(self, env_transition_fct, obs_blender, env_reward_fct, env_is_terminal_fct, possible_action_fct,
                 discount_factor, horizon):
        # todo: I do not like the use of possible actions that is a "mcts kind of object" whereas the rest is env-like

        self.env_transition_fct = env_transition_fct
        self.obs_blender = obs_blender
        self.env_reward_fct = env_reward_fct
        self.env_is_terminal_fct = env_is_terminal_fct
        self.possible_action_fct = possible_action_fct
        self.discount_factor = discount_factor
        self.horizon = horizon

    def __call__(self, node):
        time_step = 0
        ret = 0
        powered_discount_factor = 1
        state = node.state
        obs_type = state[0]

        # if we are not yet at a blended obs we must go there in order to start the rollout
        if obs_type == 'gw_obs':
            gw_obs = state[1]
            channel_obs = self.possible_action_fct.sample('com_channel_action_space')
            blended_obs = self.obs_blender.blend(gw_obs=gw_obs, chnl_obs=channel_obs)
            past_action = node.parent.action
            assert node.parent.parent.state[0] == 'blended_obs'
            past_blended_obs = node.parent.parent.state[1]
            reward = self.env_reward_fct(blended_obs=past_blended_obs, action=past_action, next_blended_obs=blended_obs)
            ret = reward
            powered_discount_factor = powered_discount_factor * self.discount_factor
        elif obs_type == 'blended_obs':
            blended_obs = state[1]
            gw_obs = blended_obs._obs['gw_obs']
        else:
            raise NotImplementedError

        while (not self.env_is_terminal_fct(gw_obs)) and (time_step < self.horizon):
            # env with random action
            action = self.possible_action_fct.sample('gw_action_space')
            next_gw_obs = self.env_transition_fct(state=gw_obs, action=action)

            # com step with random com obs (in order to be able to compute a reward but does not
            # impact dynamics)
            next_channel_obs = self.possible_action_fct.sample('com_channel_action_space')
            next_blended_obs = self.obs_blender.blend(gw_obs=next_gw_obs, chnl_obs=next_channel_obs)

            # reward from blended obs
            reward = self.env_reward_fct(blended_obs=blended_obs, action=action, next_blended_obs=next_blended_obs)

            # incrementally computes return
            ret = ret + powered_discount_factor * reward

            powered_discount_factor = powered_discount_factor * self.discount_factor

            # update gw_state in order to rollout dynamics
            gw_obs = next_gw_obs
            blended_obs = next_blended_obs
            time_step += 1
        return ret


class GridworldOnlyReward(object):
    def __init__(self, env_reward):
        self.env_reward = env_reward

    def __call__(self, blended_obs, action, next_blended_obs):
        gw_obs = blended_obs._obs['gw_obs']
        next_gw_obs = next_blended_obs._obs['gw_obs']
        return self.env_reward(state=gw_obs, action=action, next_state=next_gw_obs)


class Reward(object):
    def __init__(self, reward_fct):
        self.reward_fct = reward_fct

    def __call__(self, blended_obs, action, next_blended_obs):
        obs = blended_obs._obs['blended_obs']
        return self.reward_fct(obs=obs, action=action)


class ActionMappingReward(object):

    def __call__(self, blended_obs, action, next_blended_obs):
        if blended_obs._obs['channel_obs'] == action:
            return 1
        else:
            return 0


class ActionMappingRewardWithGoalInfo(object):

    def __init__(self, distance_fct):
        self._compute_distance_to_goal = distance_fct

    def __call__(self, blended_obs, action, next_blended_obs):
        gw_obs = blended_obs._obs['gw_obs']
        next_gw_obs = next_blended_obs._obs['gw_obs']
        progress = self._compute_distance_to_goal(gw_obs) - self._compute_distance_to_goal(next_gw_obs)
        assert type(blended_obs._obs['channel_obs']) == int
        if blended_obs._obs['channel_obs'] == action and progress > 0:
            # if we followed the message and it was a good message (that could have been provided by the archi)
            return 1
        elif blended_obs._obs['channel_obs'] == action and progress == 0:
            # if we followed the message but did not move
            if self._compute_distance_to_goal(gw_obs) == 0:
                # if we were already on the goal, it was a good message to stay there
                return 1
            else:
                # if we were not on the goal, the message should have made us move closer to it
                return 0
        else:
            return 0
