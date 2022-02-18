from typing import MutableMapping

import numpy as np
from gym import spaces
from gym.utils import seeding
import torch


class VariableLenBuffer(object):
    """
    Variable length Replay Buffer
    """

    def __init__(self, keys):
        self._data = {key: [] for key in keys}
        self._nkeys = len(keys)
        self._n = 0

    def append(self, **kwargs):
        assert len(kwargs) == self._nkeys
        _ = [self._data[key].append(np.asarray(value)) for key, value in kwargs.items()]
        self._n += 1

    def extend(self, **kwargs):
        assert len(kwargs) == self._nkeys
        # makes sure we have the same number of entries
        n_transitions = set([len(value) for value in kwargs.values()])
        assert len(n_transitions) == 1
        _ = [self._data[key].extend([np.asarray(x) for x in value]) for key, value in kwargs.items()]
        self._n += n_transitions.pop()

    def __len__(self):
        return self._n

    @property
    def n_transitions(self):
        return len(self)

    def get_all_current_as_np(self):
        return {key: np.array(val) for key, val in self._data.items()}

    def clear(self):
        self.__init__(keys=list(self._data.keys()))

    def flush(self):
        """
        Returns the content of the buffer and re-initialises it
        """
        data = self.get_all_current_as_np()
        self.clear()
        return data


class ReplayBuffer(object):
    def __init__(self, capacity, obs_space, act_space, seed):

        self.capacity = capacity

        if isinstance(act_space, spaces.Discrete):
            act_dim = 1
        elif isinstance(act_space, spaces.MultiDiscrete):
            act_dim = len(act_space.nvec)
        elif isinstance(act_space, spaces.Box):
            act_dim = act_space.shape[0]
        else:
            raise NotImplementedError

        if isinstance(obs_space, spaces.MultiDiscrete):
            obs_dim = len(act_space.nvec)
        elif isinstance(obs_space, spaces.Box):
            obs_dim = obs_space.shape[0]
        else:
            raise NotImplementedError

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.seed = seed
        self.np_random, _ = seeding.np_random(seed)

        self.flush()

    def __len__(self):
        return self.filled_i

    def flush(self):
        self.states = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)

        self.filled_i = 0  # index of first empty location in buffer (last index + 1 when full)
        self.curr_i = 0  # current index to write to (overwrite oldest data)

    def store(self, state, action, reward, next_state):

        self.states[self.curr_i:self.curr_i + 1] = state
        self.next_states[self.curr_i:self.curr_i + 1] = next_state
        self.actions[self.curr_i:self.curr_i + 1] = action
        self.rewards[self.curr_i:self.curr_i + 1] = reward

        # Update the pointers
        self.curr_i = (self.curr_i + 1) % self.capacity
        self.filled_i = min(self.filled_i + 1, self.capacity)

    def sample(self, N, to_torch=True):
        assert len(self.states) == len(self.actions) == len(self.rewards) == len(self.next_states)
        if len(self.states) < N:
            inds = np.arange(0, len(self.states))
        else:
            inds = self.np_random.choice(np.arange(0, len(self.states)), N)

        batch_states = self.states[inds]
        batch_actions = self.actions[inds]
        batch_rewards = self.rewards[inds]
        batch_next_states = self.next_states[inds]

        if to_torch:
            batch_states = torch.tensor(batch_states, dtype=torch.float32)
            batch_actions = torch.tensor(batch_actions, dtype=torch.float32)
            batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)
            batch_next_states = torch.tensor(batch_next_states, dtype=torch.float32)

        return batch_states, batch_actions, batch_rewards, batch_next_states


def key_transform(key):
    if isinstance(key, np.ndarray):
        key.flags.writeable = False
        return key.tostring()
    elif isinstance(key, list):
        return tuple(key)
    else:
        return key


_sentinel = object()


class TransformDict(MutableMapping):
    __slots__ = ('_transform', '_original', '_data')

    def __init__(self, transform, init_dict=None, **kwargs):
        '''Create a new TransformDict with the given *transform* function.
        *init_dict* and *kwargs* are optional initializers, as in the
        dict constructor.
        '''
        if not callable(transform):
            raise TypeError("expected a callable, got %r" % transform.__class__)
        self._transform = transform
        # transformed => original
        self._original = {}
        self._data = {}
        if init_dict:
            self.update(init_dict)
        if kwargs:
            self.update(kwargs)

    def getitem(self, key):
        'D.getitem(key) -> (stored key, value)'
        transformed = self._transform(key)
        original = self._original[transformed]
        value = self._data[transformed]
        return original, value

    @property
    def transform_func(self):
        "This TransformDict's transformation function"
        return self._transform

    # Minimum set of methods required for MutableMapping

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._original.values())

    def __getitem__(self, key):
        return self._data[self._transform(key)]

    def __setitem__(self, key, value):
        transformed = self._transform(key)
        self._data[transformed] = value
        self._original.setdefault(transformed, key)

    def __delitem__(self, key):
        transformed = self._transform(key)
        del self._data[transformed]
        del self._original[transformed]

    # Methods overriden to mitigate the performance overhead.

    def clear(self):
        'D.clear() -> None.  Remove all items from D.'
        self._data.clear()
        self._original.clear()

    def __contains__(self, key):
        return self._transform(key) in self._data

    def get(self, key, default=None):
        'D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.'
        return self._data.get(self._transform(key), default)

    def pop(self, key, default=_sentinel):
        '''D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
          If key is not found, d is returned if given, otherwise KeyError is raised.
        '''
        transformed = self._transform(key)
        if default is _sentinel:
            del self._original[transformed]
            return self._data.pop(transformed)
        else:
            self._original.pop(transformed, None)
            return self._data.pop(transformed, default)

    def popitem(self):
        '''D.popitem() -> (k, v), remove and return some (key, value) pair
           as a 2-tuple; but raise KeyError if D is empty.
        '''
        transformed, value = self._data.popitem()
        return self._original.pop(transformed), value

    # Other methods

    def copy(self):
        'D.copy() -> a shallow copy of D'
        other = self.__class__(self._transform)
        other._original = self._original.copy()
        other._data = self._data.copy()
        return other

    __copy__ = copy

    def __getstate__(self):
        return (self._transform, self._data, self._original)

    def __setstate__(self, state):
        self._transform, self._data, self._original = state

    def __repr__(self):
        try:
            equiv = dict(self)
        except TypeError:
            # Some keys are unhashable, fall back on .items()
            equiv = list(self.items())
        return '%s(%r, %s)' % (self.__class__.__name__,
                               self._transform, repr(equiv))