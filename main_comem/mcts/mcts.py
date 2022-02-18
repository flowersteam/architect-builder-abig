from main_comem.mcts.node import ChanceNode, DecisionNode


class MCTS(object):
    """
    A Generic MCTS object that can take ANY transition, reward func and state representation

    Parameters
    ----------
    state_cast_fct : Extract features from states (cast to tuple in standard MCTS or blend if two stages).
    transition_fct : The environment transition function used to compute next_state from action.
    reward_fct : The reward function used to evaluate immediate state reward during a transition
    is_terminal_fct : Check if state is terminal.
    possible_actions_fct : Function that outputs possible actions from a given state.
    budget : Budget used for the search (Number of trajectories used to create the tree).
    tree_policy : Policy used during expansion.
    default_policy : Policy used during simulation to compute node value.
    discount_factor : Discount factor used to compute node value.
    keep_tree : Boolean, if True we keep tree if current state is a node that has already been searched.
    get_new_root : Function that computes the new root from state and picked action if keep_tree is True.
    max_depth : Max depth of the Tree.
    seed : Seed
    """

    def __init__(self, state_cast_fct, transition_fct, reward_fct, is_terminal_fct, possible_actions_fct,
                 budget, tree_policy, default_policy, discount_factor, keep_tree, get_new_root, max_depth):

        # env specific quantities
        self.state_cast_fct = state_cast_fct
        self.transition_fct = transition_fct
        self.reward_fct = reward_fct
        self.is_terminal_fct = is_terminal_fct
        self.possible_actions_fct = possible_actions_fct

        # search specific quantities
        self.budget = budget
        self.tree_policy = tree_policy
        self.default_policy = default_policy
        self.discount_factor = discount_factor

        self.get_new_root = get_new_root
        self.keep_tree = keep_tree
        self.max_depth = max_depth
        self.tree = None
        # Note that MCTS does not define any internal random process so it does not need to be seeded

    def act(self, state):

        # todo: I think that this should be removed and the hash property be handled by the nodes when
        # they create their dicts.
        state = self.state_cast_fct(state)
        # we use dictionaries so states must be hashable.
        # Anyhow it makes no sense to have mutable states in an MCTS
        assert hasattr(state, '__hash__')

        # Initialize root node of the search
        #

        if self.keep_tree:
            if self.tree is None:
                # if we do not have a tree already we create one
                root_node = DecisionNode(parent=None, state=state, is_terminal=self.is_terminal_fct(state),
                                         possible_actions=self.possible_actions_fct(state))
                self.tree = Tree(root_node)
            else:
                # if we already have a tree
                # we compute the action that we supposedly took at the last time step. Here

                picked_action = self.tree.last_action

                existing_decision_node = self.get_new_root(state=state,
                                                           root_node=self.tree.root,
                                                           picked_action=picked_action)
                # if sampled state corresponds to an existing decision node
                if existing_decision_node is not None:
                    # we reuse the tree by making this decision node the root of the search
                    root_node = existing_decision_node
                    root_node.parent = None

                else:
                    # we cannot reuse the tree
                    root_node = DecisionNode(parent=None, state=state, is_terminal=self.is_terminal_fct(state),
                                             possible_actions=self.possible_actions_fct(state))

                self.tree.update_root(root_node)

        else:
            root_node = DecisionNode(parent=None, state=state, is_terminal=self.is_terminal_fct(state),
                                     possible_actions=self.possible_actions_fct(state))

        # Search
        #

        for i in range(self.budget):
            rewards = []

            # apply tree-policy recursively to select a leaf-node and sample from it
            selected_leaf_node, sampled_action, sampled_next_state, rewards = self.select_and_sample(node=root_node,
                                                                                                     rewards=rewards)

            # add expanded nodes to tree (decision node (new state from known action)
            # or chance-node + decision (new action))
            new_node = self.extend_tree(leaf_node=selected_leaf_node,
                                        action=sampled_action,
                                        next_state=sampled_next_state)

            # estimates the value of new node's state with default-policy (either runs a simulation
            # or estimates a value directly)
            # Todo: I think that this is not nice because I evaluate the reward from past observations in simulation
            # This should probably be done before doing simulation
            new_node_value = self.default_policy(node=new_node)

            # update the value of the chance nodes encountered along the trajectory
            self.backpropagate(node=new_node, rewards=rewards, node_value=new_node_value)

        # Exploit search
        #
        best_action = self.pick_best_action(root_node)

        if self.tree is not None:
            self.tree.last_action = best_action

        return best_action

    def pick_best_action(self, node):
        return max(node.children, key=lambda x: node.children[x].value)

    def select_and_sample(self, node, rewards, depth=0):

        # todo: this should be seriously checked because I am unsure that it is the right way of doing it
        if node.is_terminal or depth >= self.max_depth:
            return node, None, None, rewards

        else:
            # we select an action, either from already expended ones if not a leaf-node
            # or a sample a new one is leaf-node (not fully-expanded)
            action = self.tree_policy(node)

            # state transition and associated reward
            next_state = self.transition_fct(decision_node=node, action=action)

            rew = self.reward_fct(decision_node=node, action=action, next_state=next_state)

            rewards.append(rew)

            # the chance-node corresponding to this action already exists
            if action in node.children:

                # the decision-node corresponding to this pair of (action, next-state) already exists
                if next_state in node.children[action].children:
                    # we recursively go deeper in the tree until we either try a new action
                    # or an existing action ends up in a new state
                    depth += 1
                    return self.select_and_sample(node=node.children[action].children[next_state],
                                                  rewards=rewards, depth=depth)

            # we have a new decision node to create, either because we tried a new action
            # or because an existing action ended up in a new state
            return node, action, next_state, rewards

    def extend_tree(self, leaf_node, action, next_state):

        # the leaf_node is a terminal state, there is nothing to extend
        # todo: this should be seriously checked because I am unsure that it is the right way of doing it
        if action is None or next_state is None:
            return leaf_node

        else:
            # if the chance-node corresponding to this action does not exists
            if action not in leaf_node.children:
                # we create the chance-node corresponding to this action
                leaf_node.children[action] = ChanceNode(parent=leaf_node, action=action)

            # making sure we actually have something to expand
            assert next_state not in leaf_node.children[action].children

            # we create the decision-node corresponding to the next-state from the chance-node
            leaf_node.children[action].children[next_state] = DecisionNode(parent=leaf_node.children[action],
                                                                           state=next_state,
                                                                           is_terminal=self.is_terminal_fct(next_state),
                                                                           possible_actions=
                                                                           self.possible_actions_fct(next_state))

            # we return the newly created decision-node
            return leaf_node.children[action].children[next_state]

    def backpropagate(self, node, rewards, node_value):
        assert isinstance(node, DecisionNode)

        node.visits += 1

        # while the decision node has a parent chance node
        while node.parent is not None:
            rew = rewards.pop()

            if rew is not None:
                # if no new reward is provided the node value stays the same (like for two-stage)
                node_value = rew + self.discount_factor * node_value

            # we add the return value to the chance node
            node.parent.sampled_returns.append(node_value)

            # we move on to the previous decision node
            node = node.parent.parent
            node.visits += 1

        assert len(rewards) == 0

    def reset_tree(self):
        self.tree = None
    @property
    def params(self):
        return None

    def get_params(self):
        return self.params


class Tree(object):
    def __init__(self, root_node):
        self.update_root(root_node)
        self.last_action = None

    def update_root(self, root_node):
        assert isinstance(root_node, DecisionNode)
        assert root_node.parent is None
        self.root = root_node
