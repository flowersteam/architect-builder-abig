# from main_comem.utils.ml import to_torch

import torch
import numpy as np


class ObsDict(dict):
    """
    Just a wrapper over python dict() on which calling len(obs_dict) raises an error
    In our use of dictionaries as containers for the agent's observations, we
    do not want any part of the code to do for example: batch_size = len(obs).
    This would silently use a batch_size of len(obs.keys()) and harm learning.
    """

    def __init__(self, dict):
        super().__init__()
        self.update(dict)

    def _get_from_index(self, indexes):
        return ObsDict({key: value[indexes] for key, value in self.items()})

    def __len__(self):
        observations_len_list = [len(values) for key, values in self.items()]
        common_len = set(observations_len_list)
        assert len(common_len) == 1, "observations have different len"
        return common_len.pop()

    def __getitem__(self, idx):
        return self._get_from_index(idx)

    def copy(self):
        return ObsDict({key: np.copy(value) for key, value in self.items()})

def torch_cat_obs(list_of_obs, dim=0):
    set_of_key_set = set([frozenset(obs.keys()) for obs in list_of_obs])
    assert len(set_of_key_set) == 1, 'all the obs do not have the same keys'
    common_key_set = set_of_key_set.pop()
    return ObsDict(
        {obs_name: torch.cat([obs.get(obs_name) for obs in list_of_obs], dim=dim) for obs_name in common_key_set})

def np_concat_obs(list_of_obs, axis=0):
    set_of_key_set = set([frozenset(obs.keys()) for obs in list_of_obs])
    assert len(set_of_key_set) == 1, 'all the obs do not have the same keys'
    common_key_set = set_of_key_set.pop()
    return ObsDict(
        {obs_name: np.concatenate([obs.get(obs_name) for obs in list_of_obs], axis=axis) for obs_name in common_key_set})

#
