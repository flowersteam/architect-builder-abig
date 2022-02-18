import math

import jax
import torch
from jax import jit, random
import jax.numpy as jnp
import jax.nn as jnn
import jax.experimental.optimizers as joptim
import numpy as np
from scipy.special import softmax

import torch.optim as optim

from main_comem.agents.agent_policy import TabularSoftMaxPolicy, MLPSoftMaxPolicy, CNNSoftMaxPolicy, FilmSoftMaxPolicy, \
    MLPEmbSoftMaxPolicy, MLPAttentionSoftMaxPolicy
from main_comem.utils.ml import to_torch, to_numpy, obs_to_torch


class DummyTabularBC(object):
    def __init__(self, *args, **kwargs):
        self.policy = None

    def set_policy(self, policy):
        self.policy = policy

    def act(self, obs):
        return self.policy.act(obs)

    def seed(self, seed):
        self.policy.seed(seed)


class TabularBC(object):
    def __init__(self, model_args, seed):
        self.policy = TabularSoftMaxPolicy(**model_args, seed=seed)

        # the reward table is different that the params of the policy in order to be able
        # to precompute the reward values to be more efficient
        self._reward_table = None

        # jit to speed-up, be ultra mindfull that jitted function defined like this CANNOT rely on
        # mutable attributes of self as the value of such attribute is not recomputed !!!!
        self.accuracy = jit(self.accuracy)
        self.update = jit(self.update)

    # jitted
    def accuracy(self, params, obs, target_actions):
        # actions should be ints
        assert jnp.shape(target_actions)[1] == 1
        logits = params[tuple(obs.T)]
        predicted_class = jnp.expand_dims(jnp.argmax(logits, axis=1), axis=1)
        assert jnp.shape(predicted_class) == jnp.shape(target_actions)

        return jnp.mean(predicted_class == target_actions)

    # jitted
    def update(self, params, obs, actions):
        # currently we wipe the bin_table clean and make a new one by recounting the new data

        indexes = jnp.concatenate((jnp.asarray(obs), actions), axis=1)
        indexes = tuple(indexes.T)

        params = jnp.zeros_like(params).at[indexes].add(1)
        return params

    def cast_obs_and_target_actions(self, obs, target_actions):
        obs = jnp.asarray(obs)
        target_actions = jnp.asarray(target_actions)

        # as many obs data as action data
        assert len(obs) == len(target_actions)
        # obs should be (batch, 3) because (batch,(x,y,m))
        assert len(jnp.shape(obs)) == 2
        # assert jnp.shape(obs)[1] == 3

        # actions might be (batch) because they are just ints so we make them (batch, 1)
        if len(jnp.shape(target_actions)) == 1:
            target_actions = jnp.expand_dims(target_actions, axis=1)
        else:
            assert len(jnp.shape(target_actions)) == 2
        assert jnp.shape(target_actions)[1] == 1

        return obs, target_actions

    def fit(self, obs, target_actions):

        ## Optim is in Jax
        jax_obs, jax_target_actions = self.cast_obs_and_target_actions(obs, target_actions)
        jax_params = jnp.asarray(self.policy.params)
        jax_params = self.update(params=jax_params,
                                 obs=jax_obs,
                                 actions=jax_target_actions)

        acc = self.accuracy(params=jax_params,
                            obs=jax_obs,
                            target_actions=jax_target_actions)

        ## But policy will be used as np
        self.policy.update_params(np.asarray(jax_params))
        self._reward_table = None  # we also reset the reward values as they must be recomputed
        # with the new policy values
        acc = np.asarray(acc)
        return acc

    def reward_fct(self, obs, action):
        # the RL objective with this reward minimizes the cross-entropy between expert-occupancy measure
        # and agent's occupancy measure. It is like AIRL RL objective but without the policy entropy bonus
        # thus cross-entropy instead of (reverse) KL.

        # should not be called on batch
        if not isinstance(obs, tuple):
            obs = tuple(obs)
        loglikelihoods = self.reward_table[obs]
        return loglikelihoods[action]

    @property
    def reward_table(self):
        # the reward values have not been computed yet
        if self._reward_table is None:
            self.compute_reward_table()
        return self._reward_table

    def compute_reward_table(self):
        # todo: I think this might the reason why IRL + RL does not work, because we take a 1/N approach rather
        # todo: than a softmax that we do for BC-policy. the question is how to do a softmax on the whole policy params?
        # todo: I think we should do something like:
        #
        # temp_policy_params = self.policy.params.reshape(1,-1)
        # visitation = softmax(temp_policy_params)
        # visitation = visitation.reshape(*self.policy.params.shape)
        # self._reward_table = visitation

        # we add one visit count to every s,a pairs to make sure we have full coverage and we do not have
        # 0 probability and numerical instability (in practice every the state can be initial and the policy is
        # stochastic so we should have non-zero proba everywhere
        safe_visit = self.policy.params + np.ones_like(self.policy.params)
        # reward values are s,a log-likelihood under expert demos
        self._reward_table = np.log(safe_visit / np.sum(safe_visit))

    def act(self, obs):
        return self.policy.act(obs)

    def seed(self, seed):
        self.policy.seed(seed)


class GradientDescentTabularBC(TabularBC):
    def __init__(self, model_args, seed):
        super().__init__(model_args={k: model_args.pop(k) for k in ['obs_space', 'act_space']}, seed=seed)
        self.lr = model_args['lr']
        self.max_epoch = model_args['max_epoch']
        self.batch_size = model_args['batch_size']
        self.key = random.PRNGKey(seed)

        # create optimizer
        self.opt_init, self.opt_update, self.opt_get_params = joptim.adam(self.lr)

    def cross_entropy_loss(self, params, obs, one_hot_actions):
        logits = params[tuple(obs.T)]
        log_likelihood = jnn.log_softmax(logits, axis=1)
        loss = -jnp.mean(jnp.sum(log_likelihood * one_hot_actions, axis=1), axis=0)
        return loss

    def one_hot(self, x, k, dtype=jnp.float32):
        return jnp.array(x == jnp.arange(k), dtype=dtype)

    def cast_obs_and_target_actions(self, obs, target_actions):
        obs, target_actions = super().cast_obs_and_target_actions(obs, target_actions)
        one_hot_target_actions = self.one_hot(target_actions, self.policy.act_space.n)
        return obs, target_actions, one_hot_target_actions

    # jitted
    def update(self, obs, one_hot_actions, params, step_i, opt_state):

        value, grads = jax.value_and_grad(self.cross_entropy_loss)(params,
                                                                   obs,
                                                                   one_hot_actions)
        opt_state = self.opt_update(step_i, grads, opt_state)
        return value, opt_state

    def fit(self, obs, target_actions):

        ## Optimization is done in Jax
        # cast the data to jax
        obs, target_actions, one_hot_target_actions = self.cast_obs_and_target_actions(obs, target_actions)
        n_data = len(obs)
        indexes = jnp.arange(n_data)

        # create optimizer state
        opt_state = self.opt_init(jnp.asarray(self.policy.params))

        for epoch in range(self.max_epoch):
            loss_values = []
            k, self.key = random.split(self.key)
            indexes = random.permutation(k, indexes)
            for i in range(n_data // self.batch_size):
                batch_indexes = indexes[i:i + self.batch_size]
                batch_obs = obs[batch_indexes]
                batch_one_hot_target_actions = one_hot_target_actions[batch_indexes]

                value, opt_state = self.update(obs=batch_obs,
                                               one_hot_actions=batch_one_hot_target_actions,
                                               step_i=epoch,
                                               opt_state=opt_state,
                                               params=self.opt_get_params(opt_state))
                loss_values.append(value)

            mean_loss = jnp.mean(jnp.asarray(loss_values))
            acc = self.accuracy(params=self.opt_get_params(opt_state),
                                obs=obs,
                                target_actions=target_actions)

            print(f'Epoch {epoch} --- mean-loss: {mean_loss}, accuracy: {acc}\n')

        ## After optim everything is back in numpy
        self.policy.params = np.asarray(self.opt_get_params(opt_state))

        return np.asarray(acc)

    def seed(self, seed):
        super().seed(seed)
        self.key = random.PRNGKey(seed)

    def reward_fct(self, obs, action):
        raise NotImplementedError('The logits of a BC policy learned through max-likelihood '
                                  'do not make a consistent reward')

class PytorchGradientDescentTabularBC(TabularBC, torch.nn.Module):
    def __init__(self, model_args, seed):
        super().__init__(model_args={k: model_args.pop(k) for k in ['obs_space', 'act_space']}, seed=seed)
        torch.nn.Module.__init__(self)
        self.seed(seed)
        self.lr = model_args['lr']
        self.max_epoch = model_args['max_epoch']
        self.batch_size = model_args['batch_size']
        self.split_prop = 0.7
        self.max_wait = 1000
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.temperature = model_args['temperature']

    def seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        self.policy.seed(seed)

    def fit(self, obs, target_actions):

        self.torch_params = torch.nn.Parameter(torch.tensor(np.zeros_like(self.policy.params)))

        opt = optim.SGD(params=self.parameters(), lr=self.lr)

        def act_fct(params, batch_obs):
            # return params[batch_obs, :]/self.temperature
            return params[batch_obs, :]

        obs = obs_to_torch(obs)
        obs = torch.LongTensor(obs).squeeze()
        target_actions = to_torch(target_actions).squeeze()

        # we split the data-set in two in order to have a validation metric to monitor
        n_data = len(target_actions)
        n_train = int(self.split_prop * n_data)
        shuffling_idxs = torch.randperm(n_data)
        train_idx = shuffling_idxs[0:n_train]
        valid_idx = shuffling_idxs[n_train:n_data]

        train_obs = obs[train_idx]
        train_act = target_actions[train_idx]
        n_train = len(train_act)

        valid_obs = obs[valid_idx]
        valid_act = target_actions[valid_idx]
        n_valid = len(valid_act)

        # we use this to monitor the learning
        n_wait = 1
        best_performance = float('-inf')

        n_itr = int(n_train / self.batch_size)
        assert n_itr > 0
        for epoch in range(self.max_epoch):

            # shuffle the data at each epoch
            shuffling_idxs = torch.randperm(n_train)
            train_obs = train_obs[shuffling_idxs]
            train_act = train_act[shuffling_idxs]

            # train for one epoch
            loss_rec = []
            for i in range(n_itr):
                batch_obs = train_obs[i * self.batch_size:(i + 1) * self.batch_size]
                batch_act = train_act[i * self.batch_size:(i + 1) * self.batch_size]

                logits = act_fct(self.torch_params, batch_obs)
                loss = self.loss_fct(input=logits, target=batch_act.long())

                opt.zero_grad()
                loss.backward()
                opt.step()

                loss_rec.append(to_numpy(loss))

            # compute performance after each epoch
            performance = self.get_accuracy(valid_obs, valid_act, lambda x : act_fct(self.torch_params, x))

            # check if there is still improvement
            if performance > best_performance:
                best_performance = performance
                n_wait = 0
            else:
                n_wait += 1

            print(f'Epoch {epoch}: loss = {np.mean(loss_rec)}, accuracy = {performance}, n_wait = {n_wait}, '
                  f'train_accuracy = {self.get_accuracy(train_obs, train_act, lambda x : act_fct(self.torch_params, x))}')

            if n_wait >= self.max_wait:
                break
            # if performance == 1.:
            #     break
        self.policy.params = self.torch_params.data.cpu().numpy()/self.temperature
        return performance, n_wait, epoch

    def get_accuracy(self, obs, act, act_fct):
        logits = act_fct(obs)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == act).type(torch.float).mean(0)
        return accuracy



class BC(object):
    def __init__(self, model_args, seed):
        if not 'temperature' in model_args:
            model_args['temperature'] = 1.

        self.policy = self.make_network(model_args, seed)

        self.model_args = model_args
        self.seed_val = seed
        self.lr = model_args['lr']
        self.optim = optim.Adam(params=self.policy.params, lr=self.lr)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.max_epoch = model_args['max_epoch']
        self.batch_size = model_args['batch_size']
        self.reset_optimizer = model_args['reset_optimizer']
        # resetting the network will automatically reset the optimizer
        self.reset_network = model_args['reset_network']

        # will be used to monitor learning
        self.max_wait = model_args['max_wait']
        self.split_prop = 0.7

    def make_network(self, model_args, seed):
        if model_args['network_type'] == 'mlp':
            policy = MLPSoftMaxPolicy(obs_space=model_args['obs_space'],
                                      act_space=model_args['act_space'],
                                      seed=seed,
                                      temperature=model_args['temperature'])
        elif model_args['network_type'] == 'cnn':
            policy = CNNSoftMaxPolicy(obs_space=model_args['obs_space'],
                                      act_space=model_args['act_space'],
                                      seed=seed,
                                      temperature=model_args['temperature'])
        elif model_args['network_type'] == 'film':
            policy = FilmSoftMaxPolicy(obs_space=model_args['obs_space'],
                                       act_space=model_args['act_space'],
                                       seed=seed,
                                       temperature=model_args['temperature'])
        elif model_args['network_type'] == 'mlp_emb':
            policy = MLPEmbSoftMaxPolicy(obs_space=model_args['obs_space'],
                                         act_space=model_args['act_space'],
                                         seed=seed,
                                         temperature=model_args['temperature'])
        elif model_args['network_type']== 'mlp_attention':
            policy = MLPAttentionSoftMaxPolicy(obs_space=model_args['obs_space'],
                                         act_space=model_args['act_space'],
                                         seed=seed,
                                         temperature=model_args['temperature'])
        else:
            raise NotImplementedError

        return policy

    def fit(self, obs, target_actions):

        if self.reset_network:
            # be carefull that this line may break links between models (some policies might still link to previous network)
            self.policy = self.make_network(self.model_args, self.seed_val)
            self.optim = optim.Adam(params=self.policy.params, lr=self.lr)
        elif self.reset_optimizer:
            self.optim = optim.Adam(params=self.policy.params, lr=self.lr)

        obs = obs_to_torch(obs)
        target_actions = to_torch(target_actions)

        # we split the data-set in two in order to have a validation metric to monitor
        n_data = len(target_actions)
        n_train = int(self.split_prop * n_data)
        shuffling_idxs = torch.randperm(n_data)
        train_idx = shuffling_idxs[0:n_train]
        valid_idx = shuffling_idxs[n_train:n_data]

        train_obs = obs[train_idx]
        train_act = target_actions[train_idx]
        n_train = len(train_act)

        valid_obs = obs[valid_idx]
        valid_act = target_actions[valid_idx]
        n_valid = len(valid_act)

        # we use this to monitor the learning
        n_wait = 1
        best_performance = float('-inf')

        n_itr = int(n_train / self.batch_size)
        assert n_itr > 0
        for epoch in range(self.max_epoch):

            # shuffle the data at each epoch
            shuffling_idxs = torch.randperm(n_train)
            train_obs = train_obs[shuffling_idxs]
            train_act = train_act[shuffling_idxs]

            # train for one epoch
            loss_rec = []
            for i in range(n_itr):
                batch_obs = train_obs[i * self.batch_size:(i + 1) * self.batch_size]
                batch_act = train_act[i * self.batch_size:(i + 1) * self.batch_size]

                logits = self.policy(batch_obs)
                loss = self.loss_fct(input=logits, target=batch_act.long())

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                loss_rec.append(to_numpy(loss))

            # compute performance after each epoch
            performance = self.get_accuracy(valid_obs, valid_act)

            # check if there is still improvement
            if performance > best_performance:
                best_performance = performance
                n_wait = 0
            else:
                n_wait += 1

            print(f'Epoch {epoch}: loss = {np.mean(loss_rec)}, accuracy = {performance}, n_wait = {n_wait}, '
                  f'train_accuracy = {self.get_accuracy(train_obs, train_act)}')

            if n_wait >= self.max_wait:
                break
            # if performance == 1.:
            #     break

        return performance, n_wait, epoch

    def act(self, obs):
        return self.policy.act(obs)

    def get_accuracy(self, obs, act):
        logits = self.policy(obs)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == act).type(torch.float).mean(0)
        return accuracy

    def seed(self, seed):
        self.policy.seed(seed)

    def reward_fct(self, obs, action):
        raise NotImplementedError('The logits of a BC policy learned through max-likelihood '
                                  'do not make a consistent reward')
