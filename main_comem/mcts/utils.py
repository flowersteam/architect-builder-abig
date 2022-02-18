from gym import spaces
from gym.utils import seeding

from main_comem.mcts.node import DecisionNode


def get_new_root(state, root_node, picked_action):
    picked_chance_node = root_node.children[picked_action]
    if state in picked_chance_node.children:
        return picked_chance_node.children[state]
    else:
        return None


class PossibleActions(object):
    def __init__(self, action_space, seed):
        assert isinstance(action_space, spaces.Discrete)
        self.action_space = action_space
        self.np_random, _ = seeding.np_random(None)
        self.seed(seed)

    def seed(self, seed):
        self.np_random.seed(seed)

    def __call__(self, state=None):
        possible_actions = list(range(self.action_space.n))
        self.np_random.shuffle(possible_actions)
        return possible_actions

    def sample(self, state=None):
        return self.action_space.sample()


class EnvTransitionFct(object):
    def __init__(self, env_transition_fct):
        self.env_transition_fct = env_transition_fct

    def __call__(self, decision_node, action):
        assert isinstance(decision_node, DecisionNode)
        state = decision_node.state

        next_state = self.env_transition_fct(state, action)
        return next_state


class EnvIsTerminalFct(object):
    def __init__(self, env_is_terminal_fct):
        self.env_is_terminal_fct = env_is_terminal_fct

    def __call__(self, state):
        return self.env_is_terminal_fct(state)


class EnvRewardFct(object):
    def __init__(self, env_reward_fct):
        self.env_reward_fct = env_reward_fct

    def __call__(self, decision_node, action, next_state):
        state = decision_node.state
        rew = self.env_reward_fct(state, action, next_state)
        return rew