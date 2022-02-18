from main_comem.utils.data_structures import key_transform, TransformDict


class DecisionNode(object):
    """
    Decision node class, labelled by a state
    """

    def __init__(self, parent, state, possible_actions, is_terminal):
        self.parent = parent
        self.state = state
        self.is_terminal = is_terminal

        self.children = TransformDict(
            transform=key_transform)  # dict of children chance-nodes, keys are the actions of the chance nodes

        self.unsampled_actions = possible_actions
        self.visits = 0


class ChanceNode(object):
    """
    Chance node class, labelled by a state-action pair
    The state is accessed via the parent attribute
    """

    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = TransformDict(
            transform=key_transform)  # dict of children decision nodes, keys are the states of the child nodes
        self.sampled_returns = []

    @property
    def value(self):
        return sum(self.sampled_returns) / len(self.sampled_returns)

    @property
    def n(self):
        return len(self.sampled_returns)
