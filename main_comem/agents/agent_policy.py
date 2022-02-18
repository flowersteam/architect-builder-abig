from gym.spaces import Discrete, MultiDiscrete, Box
from gym.utils import seeding
from scipy.special import softmax
import numpy as np
import torch
import torch.nn.functional as F

from env_comem.gym_gridworld.gridworld import ACTIONS_DIRECTION_DICT, PRINT_ACTION_DICT
from env_comem.blender import TabularBlendedObs

from main_comem.utils.networks import MLPNetwork, ConvNetwork, FiLmCNNNetwork, MLPWordEmbeddingNetwork, MLPAttentionNetwork
from main_comem.utils.ml import to_numpy, obs_to_torch
from main_comem.utils.data_structures import TransformDict, key_transform


class Policy(object):
    def __init__(self, obs_space, act_space):
        self.obs_space = obs_space
        self.act_space = act_space

    def act(self, **kwargs):
        raise NotImplementedError

    def seed(self, seed):
        return NotImplementedError


class RandomTabularPolicy(Policy):
    def __init__(self, obs_space, act_space):
        super().__init__(obs_space, act_space)
        assert isinstance(act_space, Discrete)

    def act(self, *args, **kwargs):
        return self.act_space.sample()

    def get_params(self):
        return None


class OptimalGWPolicy(Policy):
    def __init__(self, obs_space, act_space, gw):
        super().__init__(obs_space, act_space)
        assert isinstance(act_space, Discrete)

        self.gw = gw

        # we revert the action to direction dict
        self.direction_action_dict = {value: key for key, value in ACTIONS_DIRECTION_DICT.items()}

    def act(self, state):
        goal_state = self.gw.encode_obs(self.gw._goal_state, self.gw.obs_type)

        diff_vec = np.asarray(goal_state) - np.asarray(state)
        if all(diff_vec == np.asarray([0, 0])):
            direction = diff_vec
        else:
            principal_dir = np.argmax(np.abs(diff_vec))
            direction = np.zeros_like(diff_vec)
            direction[principal_dir] = np.sign(diff_vec[principal_dir])

        action = self.direction_action_dict[tuple(direction)]
        return action


class ActionToMessage(object):
    def __init__(self):
        self.mapping = lambda x: x

    def map(self, action):
        assert isinstance(action, int)
        return self.mapping(action)


class StateConditionedMessageToActionBijectiveMapping(Policy):
    # Deterministic policy, for each state, each message corresponds to a distinct
    # action. For each state, which message correspond to which action is randomly
    # defined at init
    def __init__(self, obs_space, act_space):
        super().__init__(obs_space, act_space)
        assert isinstance(act_space, Discrete)
        assert isinstance(obs_space, MultiDiscrete)

        # we need as many actions as message to create a bijective mapping
        assert act_space.n == obs_space.nvec[-1]

        # action = mapping(x,y,m)

        self.mapping = np.zeros(obs_space.nvec, dtype=np.int)

        for x in range(obs_space.nvec[0]):
            for y in range(obs_space.nvec[1]):
                actions = self.act_space.np_random.permutation(self.act_space.n)
                self.mapping[x, y, :] = actions[:obs_space.nvec[2]]

    def act(self, state):

        if isinstance(state, TabularBlendedObs):
            state = tuple(state.state)

        return self.mapping[state]

    @property
    def params(self):
        table_shape = np.concatenate([self.mapping.shape, [self.act_space.n]])
        one_hots = np.zeros(table_shape).reshape(-1, table_shape[-1])
        idx = self.mapping.flatten()
        one_hots[np.arange(len(idx)), idx] = 1
        one_hots = one_hots.reshape(table_shape)
        return one_hots

    def get_params(self):
        return self.params


class StateConditionedMessageToActionInjectiveMapping(Policy):
    # Deterministic policy, for each state, each message corresponds to an action and
    # different messages can lead to the same action. Thus in some states, some actions
    # might be unavailable.
    # For each state, which message correspond to which action is randomly
    # defined at init
    def __init__(self, obs_space, act_space):
        super().__init__(obs_space, act_space)
        assert isinstance(act_space, Discrete)
        assert isinstance(obs_space, MultiDiscrete)

        # action = mapping(x,y,m)
        self.mapping = np.zeros(obs_space.nvec, dtype=np.int)

        for x in range(obs_space.nvec[0]):
            for y in range(obs_space.nvec[1]):
                actions = self.act_space.np_random.randint(self.act_space.n, size=obs_space.nvec[2])
                self.mapping[x, y, :] = actions

    def act(self, state):

        if isinstance(state, TabularBlendedObs):
            state = tuple(state.state)

        return self.mapping[state]

    @property
    def params(self):
        table_shape = np.concatenate([self.mapping.shape, [self.act_space.n]])
        one_hots = np.zeros(table_shape).reshape(-1, table_shape[-1])
        idx = self.mapping.flatten()
        one_hots[np.arange(len(idx)), idx] = 1
        one_hots = one_hots.reshape(table_shape)
        return one_hots

    def get_params(self):
        return self.params


class StateConditionedMessageToActionStochasticMapping(Policy):
    # Stochastic policy, for each state and message the action is sampled from a categorical
    # distribution. For each state and message the probability of each action is defined from
    # a Dirichlet distribution at init
    def __init__(self, obs_space, act_space, concentration_param):
        super().__init__(obs_space, act_space)
        assert isinstance(act_space, Discrete)
        assert isinstance(obs_space, MultiDiscrete)

        # Dirichlet distribution concentration params (kind of inverse of temperature of the softmax)
        # if close to 0, only spiky distributions are likely
        # if close to 1, all distribution are likely
        # if close to inf, only uniform distribution are likely
        self.concentration_param = concentration_param

        # proba = mapping(x,y,m,a)
        self.probas = np.zeros(list(obs_space.nvec) + [self.act_space.n])

        for x in range(obs_space.nvec[0]):
            for y in range(obs_space.nvec[1]):
                for m in range(obs_space.nvec[2]):
                    probas = self.act_space.np_random.dirichlet(alpha=[concentration_param] * self.act_space.n)
                    self.probas[x, y, m, :] = probas

    def act(self, state):

        if isinstance(state, TabularBlendedObs):
            state = tuple(state.state)

        return int(self.act_space.np_random.choice(np.arange(self.act_space.n), p=self.probas[tuple(state)]))

    @property
    def params(self):
        return self.probas

    def get_params(self):
        return self.probas

    @params.setter
    def params(self, value):
        self.probas = value


class TabularSoftMaxPolicy(Policy):
    def __init__(self, obs_space, act_space, seed):
        super().__init__(obs_space, act_space)

        assert isinstance(obs_space, MultiDiscrete)

        assert (isinstance(act_space, Discrete))

        self.np_random, _ = seeding.np_random(seed)

        # the table is going to be of shape (n_obs*, n_actions) and
        # give the logit such that softmax(params[n_obs*], axis=0) gives the proba of each action a
        # in state obs
        self.table_dims = list(obs_space.nvec) + [act_space.n]
        self.params = np.zeros(self.table_dims)

    def update_params(self, new_params):
        assert new_params.shape == self.params.shape
        self.params = new_params

    def act(self, obs):
        if not isinstance(obs, tuple):
            if isinstance(obs, TabularBlendedObs):
                obs = obs.state
            else:
                obs = tuple(obs)

        assert len(np.shape(obs)) == 1
        logits = self.params[obs]
        # todo: actually our method might not work if we use a proportional probability instead of a softmax one
        probas = softmax(logits, axis=0)
        return self.np_random.choice(np.arange(self.act_space.n), p=probas)

    def seed(self, seed):
        self.np_random, _ = seeding.np_random(seed)

    def print_most_likely_action(self):
        table = self.params
        probas = softmax(self.params, axis=-1)
        most_likely_action_per_xym = np.argmax(probas, axis=-1)

        for m in range(most_likely_action_per_xym.shape[2]):
            print(f"Word {m}: ")
            self.print_action_matrix(most_likely_action_per_xym[:, :, m], PRINT_ACTION_DICT)
            print("-----------------")

    def print_action_matrix(self, matrix, action_dict):
        to_print = ""
        for y in reversed(range(matrix.shape[1])):
            for x in range(matrix.shape[0]):
                to_print += f"{action_dict[int(matrix[x, y])]} "
            to_print += f"\n"
        to_print = to_print[:-1]
        print(to_print)

    def get_params(self):
        return self.params


class MLPSoftMaxPolicy(Policy):
    def __init__(self, obs_space, act_space, seed, temperature=1.):
        super().__init__(obs_space, act_space)

        assert isinstance(obs_space, Box)

        assert (isinstance(act_space, Discrete))

        self.np_random, _ = seeding.np_random(seed)
        self.temperature = temperature

        self.mlp = MLPNetwork(num_inputs=np.prod(self.obs_space.shape), num_outputs=self.act_space.n,
                              hidden_size=126, set_final_bias=False)
        self.deterministic = False

    def act(self, obs):
        assert len(np.shape(obs)) == 1
        obs = obs_to_torch(obs, unsqueeze_dim=0)
        logits = self(obs)
        probas = to_numpy(F.softmax(logits, dim=1).squeeze())
        # print(probas)

        if self.deterministic:
            val = np.argmax(probas)
            return val
        else:
            return self.np_random.choice(np.arange(self.act_space.n), p=probas)

    def __call__(self, batch_obs):
        return self.mlp(batch_obs) / self.temperature

    def seed(self, seed):
        self.np_random, _ = seeding.np_random(seed)

    def make_deterministic(self):
        self.deterministic = True

    @property
    def params(self):
        return self.mlp.parameters()

    def get_params(self):
        return self.mlp.state_dict()


class CNNSoftMaxPolicy(Policy):
    # todo: currently we are not leveraging GPU ! should be implemented with setting the correct devices at the right
    # moment
    def __init__(self, obs_space, act_space, seed, temperature=1.):
        super().__init__(obs_space, act_space)

        assert isinstance(obs_space, Box)

        assert (isinstance(act_space, Discrete))

        self.np_random, _ = seeding.np_random(seed)
        self.temperature = temperature

        self.cnn = ConvNetwork(input_size=obs_space.shape[:-1], num_input_channels=obs_space.shape[-1],
                               output_vec_size=act_space.n)

    def act(self, obs):
        assert len(np.shape(obs)) == 3
        obs = obs_to_torch(obs, unsqueeze_dim=0)
        logits = self(obs)
        probas = to_numpy(F.softmax(logits, dim=1).squeeze())
        # print(probas)
        return self.np_random.choice(np.arange(self.act_space.n), p=probas)

    def __call__(self, batch_obs):
        return self.cnn(batch_obs) / self.temperature

    def seed(self, seed):
        self.np_random, _ = seeding.np_random(seed)

    @property
    def params(self):
        return self.cnn.parameters()

    def get_params(self):
        return self.cnn.state_dict()


class FilmSoftMaxPolicy(Policy):
    # todo: currently we are not leveraging GPU ! should be implemented with setting the correct devices at the right
    # moment
    def __init__(self, obs_space, act_space, seed, temperature=1.):
        super().__init__(obs_space, act_space)

        assert isinstance(obs_space['tile'], Box)
        assert isinstance(obs_space['message'], Discrete)

        assert (isinstance(act_space, Discrete))

        self.np_random, _ = seeding.np_random(seed)
        self.temperature = temperature

        self.film_cnn = FiLmCNNNetwork(obs_space, act_space)

    def act(self, obs):
        obs = obs_to_torch(obs, unsqueeze_dim=0)
        logits = self(obs)
        probas = to_numpy(F.softmax(logits, dim=1).squeeze())
        # print(probas)
        return self.np_random.choice(np.arange(self.act_space.n), p=probas)

    def __call__(self, batch_obs):
        return self.film_cnn(batch_obs) / self.temperature

    def seed(self, seed):
        self.np_random, _ = seeding.np_random(seed)

    @property
    def params(self):
        return self.film_cnn.parameters()

    def get_params(self):
        return self.film_cnn.state_dict()

class MLPEmbSoftMaxPolicy(Policy):
    def __init__(self, obs_space, act_space, seed, temperature=1.):
        super().__init__(obs_space, act_space)

        assert isinstance(obs_space['tile'], Box)
        assert isinstance(obs_space['message'], Discrete)

        assert (isinstance(act_space, Discrete))

        self.np_random, _ = seeding.np_random(seed)
        self.temperature = temperature

        self.mlp_emb = MLPWordEmbeddingNetwork(obs_space, act_space)

    def act(self, obs):
        obs = obs_to_torch(obs, unsqueeze_dim=0)
        logits = self(obs)
        probas = to_numpy(F.softmax(logits, dim=1).squeeze())
        # print(probas)
        return self.np_random.choice(np.arange(self.act_space.n), p=probas)

    def __call__(self, batch_obs):
        return self.mlp_emb(batch_obs) / self.temperature

    def seed(self, seed):
        self.np_random, _ = seeding.np_random(seed)

    @property
    def params(self):
        return self.mlp_emb.parameters()

    def get_params(self):
        return self.mlp_emb.state_dict()


class MLPAttentionSoftMaxPolicy(Policy):
    def __init__(self, obs_space, act_space, seed, temperature=1.):
        super().__init__(obs_space, act_space)

        assert isinstance(obs_space['tile'], Box)
        assert isinstance(obs_space['message'], Discrete)

        assert (isinstance(act_space, Discrete))

        self.np_random, _ = seeding.np_random(seed)
        self.temperature = temperature

        self.mlp_attention = MLPAttentionNetwork(obs_space, act_space)

    def act(self, obs):
        obs = obs_to_torch(obs, unsqueeze_dim=0)
        logits = self(obs)
        probas = to_numpy(F.softmax(logits, dim=1).squeeze())
        # print(probas)
        return self.np_random.choice(np.arange(self.act_space.n), p=probas)

    def __call__(self, batch_obs):
        return self.mlp_attention(batch_obs) / self.temperature

    def seed(self, seed):
        self.np_random, _ = seeding.np_random(seed)

    @property
    def params(self):
        return self.mlp_attention.parameters()

    def get_params(self):
        return self.mlp_attention.state_dict()


def get_obs_from_measurement_set(measurement_set, obs_blender, chnl):
    bw_obs = measurement_set['states']
    dict_size = measurement_set['dict_size']
    chnl_obs = np.asarray([chnl._transform(m, chnl._type) for m in range(dict_size)])

    if len(chnl_obs.shape) == 1:
        chnl_obs = chnl_obs.reshape(dict_size, 1)

    chnl_obs = np.tile(chnl_obs, (len(bw_obs), 1))
    bw_obs = np.repeat(bw_obs, dict_size, axis=0)

    obs = obs_blender.np_batch_blend_raw(bw_obs, chnl_obs)

    if isinstance(obs, dict) and obs.get('message').shape[1] == 1:
        obs.update({'message': obs.get('message').squeeze()})
    return obs, bw_obs, chnl_obs


def compute_policy_entropy(policy, measurement_set, obs_blender, chnl):
    batch_size = 256

    obs, _, _ = get_obs_from_measurement_set(measurement_set, obs_blender, chnl)

    with torch.no_grad():
        obs = obs_to_torch(obs)

        summed_batch_entropy = 0.

        for i in range(len(obs) // batch_size + 1):
            sup = min((i + 1) * batch_size, len(obs))
            batch_obs = obs[i * batch_size: sup]

            batch_actions_proba = policy(batch_obs)
            batch_entropy = -(softmax(batch_actions_proba, axis=1) * torch.log_softmax(batch_actions_proba, dim=1)).sum(
                1)

            summed_batch_entropy += batch_entropy.sum(0)

        N = len(obs)
    return to_numpy(summed_batch_entropy / N)


def compute_accuracy_between_policies(reference_policy, policy, measurement_set, obs_blender, chnl):
    batch_size = 256

    obs, _, _ = get_obs_from_measurement_set(measurement_set, obs_blender, chnl)

    with torch.no_grad():
        obs = obs_to_torch(obs)

        summed_batch_score = 0.

        for i in range(len(obs) // batch_size + 1):
            sup = min((i + 1) * batch_size, len(obs))
            batch_obs = obs[i * batch_size: sup]

            batch_reference_logits = reference_policy(batch_obs)
            batch_reference = torch.argmax(batch_reference_logits, dim=1)

            batch_logits = policy(batch_obs)
            batch_prediction = torch.argmax(batch_logits, dim=1)

            batch_score = (batch_prediction == batch_reference).type(torch.float)

            summed_batch_score += batch_score.sum(0)

        N = len(obs)
    return to_numpy(summed_batch_score / N)


def compute_transitions_probas(policy, measurement_set, obs_blender, chnl):
    batch_size = 256

    obs, bw_obs, chnl_obs = get_obs_from_measurement_set(measurement_set, obs_blender, chnl)

    with torch.no_grad():
        obs = obs_to_torch(obs)
        probas = []
        preferred_action = []

        for i in range(len(obs) // batch_size + 1):
            sup = min((i + 1) * batch_size, len(obs))
            batch_obs = obs[i * batch_size: sup]
            batch_logits = policy(batch_obs)
            batch_probas = torch.softmax(batch_logits, dim=1)
            batch_preferred = torch.argmax(batch_logits, dim=1)

            probas.append(batch_probas)
            preferred_action.append(batch_preferred)

        probas = to_numpy(torch.cat(probas, dim=0))
        preferred_action = to_numpy(torch.cat(preferred_action, dim=0))

        nums = {'states': TransformDict(key_transform),
                'messages': TransformDict(key_transform),
                'preferred_action': TransformDict(key_transform),
                'dict_size': measurement_set['dict_size']}

        P_a_bar_sm = TransformDict(key_transform)
        P_m_bar_s = TransformDict(key_transform)  # P(m|s) = num(m|s) / sum_m(num(m|s))

        for s, m, proba, preferred_a in zip(bw_obs, chnl_obs, probas, preferred_action):
            if s not in P_m_bar_s:
                P_m_bar_s[s] = TransformDict(key_transform)

            nums['states'][s] = nums['states'].get(s, default=0) + 1
            nums['messages'][m] = nums['messages'].get(m, default=0) + 1
            nums['preferred_action'][preferred_a] = nums['preferred_action'].get(preferred_a, default=0) + 1
            P_m_bar_s[s][m] = P_m_bar_s[s].get(m, default=0) + 1

            if s not in P_a_bar_sm:
                P_a_bar_sm[s] = TransformDict(key_transform)

            if m in P_a_bar_sm[s]:
                assert np.sum(P_a_bar_sm[s][m]['probas'] - proba)/np.sum(proba) < 1e-6
                assert np.argmax(P_a_bar_sm[s][m]['preferred_action']) == preferred_a
            else:
                id = preferred_a
                P_a_bar_sm[s][m] = {'probas': proba,
                                    'preferred_action': np.asarray([1. if j == id else 0. for j in range(len(proba))])}

        for s in P_m_bar_s.keys():
            sum_on_mess = sum(list(P_m_bar_s[s].values()))
            for m in P_m_bar_s[s].keys():
                P_m_bar_s[s][m] = P_m_bar_s[s][m] / sum_on_mess

    totals = {}
    totals['states'] = sum(list(nums['states'].values()))
    assert totals['states'] == len(obs)

    totals['messages'] = sum(list(nums['messages'].values()))
    assert totals['messages'] == len(obs)

    totals['preferred_action'] = sum(list(nums['preferred_action'].values()))
    assert totals['preferred_action'] == len(obs)

    return nums, P_a_bar_sm, totals, P_m_bar_s


def compute_preferred_action_entropy(nums, P_a_bar_sm, totals):
    n_possible_actions = len(next(iter(next(iter(P_a_bar_sm.values())).values()))['probas'])

    # here we add a 1 to each unused action to avoid division by 0
    n_pa = np.asarray([nums['preferred_action'].get(a, default=1e-4) for a in range(n_possible_actions)])
    pa_dist = n_pa / sum(n_pa)
    entropy = - np.sum(pa_dist * np.log(pa_dist))

    return entropy


def compute_MIs(nums, P_a_bar_sm, P_m_bar_s, totals):
    # todo: call this in main and test it

    Ps = TransformDict(key_transform)
    for s in nums['states'].keys():
        Ps[s] = nums['states'][s] / totals['states']

    Pm = TransformDict(key_transform)
    for m in nums['messages'].keys():
        Pm[m] = nums['messages'][m] / totals['messages']

    Psm = TransformDict(key_transform)  # P(s,m) = P(s) * P(m|s)
    for s in nums['states'].keys():
        Psm[s] = TransformDict(key_transform)
        for m in P_m_bar_s[s].keys():
            Psm[s][m] = Ps[s] * P_m_bar_s[s][m]

    Psma = TransformDict(key_transform)  # P(s,m,a) = P(s,m) * P(a|s,m)

    for s in Psm.keys():
        Psma[s] = TransformDict(key_transform)
        for m in Psm[s].keys():
            Psma[s][m] = {k: v * Psm[s][m] for k, v in P_a_bar_sm[s][m].items()}

    Pa = {'probas': 0., 'preferred_action': 0.} # P(a) = sum_(s,m) P(s,m)*P(a|s,m) = sum_(s,m) P(s,m,a)
    for s in P_a_bar_sm.keys():
        for m in P_a_bar_sm[s].keys():
            Pa = {k: Pa[k] + v * Psm[s][m] for k, v in P_a_bar_sm[s][m].items()}

    Isma = {'probas': 0.,
            'preferred_action': 0.}

    for s in Psma.keys():
        for m in Psma[s].keys():
            Isma = {k: Isma[k] + v * np.log(v / (Psm[s][m] * (Pa[k] + 1e-7)) + 1e-7) for k, v in Psma[s][m].items()}
    Isma = {k: sum(v) for k, v in Isma.items()}

    Psa = TransformDict(key_transform)

    for s in Psma.keys():
        Psa[s] = {'probas': 0., 'preferred_action': 0.}
        for m in Psma[s].keys():
            Psa[s] = {k: Psa[s][k] + v for k, v in Psma[s][m].items()}

    Isa = {'probas': 0.,
           'preferred_action': 0.}

    for s in Psa.keys():
        Isa = {k: Isa[k] + v * np.log(v / (Ps[s] * (Pa[k] + 1e-7)) + 1e-7) for k, v in Psa[s].items()}

    Isa = {k: sum(v) for k, v in Isa.items()}

    Pma = TransformDict(key_transform)
    for m in Pm.keys():
        Pma[m] = {'probas': 0., 'preferred_action': 0.}
        # this assumes that all pairs sm exist for every s and every m
        for s in Ps.keys():
            Pma[m] = {k: Pma[m][k] + v for k, v in Psma[s][m].items()}

    Ima = {'probas': 0.,
           'preferred_action': 0.}

    for m in Pma.keys():
        Ima = {k: Ima[k] + v * np.log(v / (Pm[m] * (Pa[k] + 1e-7)) + 1e-7) for k, v in Pma[m].items()}

    Ima = {k: sum(v) for k, v in Ima.items()}
    return Isma['probas'], Isma['preferred_action'], Isa['probas'], Isa['preferred_action'], \
           Ima['probas'], Ima['preferred_action']
