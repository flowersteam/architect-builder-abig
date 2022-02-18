from gym.utils import seeding
from math import log

from main_comem.mcts.node import DecisionNode


class UCT(object):
    """
    Object that computes the UCT value of a node

    Parameters
    ----------
    cp : The exploration parameter of UCT
    seed : Seed
    """

    def __init__(self, cp, seed):

        self.cp = cp
        self.np_random, _ = seeding.np_random(None)
        self.seed(seed)

    def seed(self, seed):
        self.np_random.seed(seed)

    def __call__(self, node):
        assert isinstance(node, DecisionNode)

        if len(node.unsampled_actions) > 0:
            return node.unsampled_actions.pop()

        else:
            children_ucb = {key: child.value + 2 * self.cp * (log(node.visits) / child.n)**0.5
                            for key, child in node.children.items()}

            return self.rand_max(children_ucb, key=children_ucb.get)

    def rand_max(self, iterable, key=None):
        """
        A max function that tie breaks randomly instead of first-wins as in
        built-in max().
        :param iterable: The container to take the max from
        :param key: A function to compute tha max from. E.g.:
            rand_max([-2, 1], key=lambda x:x**2
          -2
          If key is None the identity is used.
        :return: The entry of the iterable which has the maximum value. Tie
        breaks are random.
        """
        if key is None:
            key = lambda x: x

        max_v = -float('inf')
        max_l = []

        for item, value in zip(iterable, [key(i) for i in iterable]):
            if value == max_v:
                max_l.append(item)
            elif value > max_v:
                max_l = [item]
                max_v = value

        return self.np_random.choice(max_l)
