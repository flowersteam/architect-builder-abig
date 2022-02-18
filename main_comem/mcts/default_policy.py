class MonteCarloReturnPolicy(object):
    """
    Monte Carlo return policy used for simulation during search

    Parameters
    ----------
    env_transition_fct : The environment transition function used to compute next_state from action.
    env_reward_fct : The reward function used to evaluate immediate state reward during a transition
    env_is_terminal_fct : Check if state is terminal.
    possible_action_fct : Function that outputs possible actions from a given state.
    discount_factor :  Discount factor used to compute node value.
    horizon : Number of time steps to run default policy
    """

    def __init__(self, env_transition_fct, env_reward_fct, env_is_terminal_fct, possible_action_fct, discount_factor,
                 horizon):
        self.env_transition_fct = env_transition_fct
        self.env_reward_fct = env_reward_fct
        self.env_is_terminal_fct = env_is_terminal_fct
        self.possible_action_fct = possible_action_fct
        self.discount_factor = discount_factor
        self.horizon = horizon

    def __call__(self, node):
        state = node.state
        time_step = 0
        ret = 0
        powered_discount_factor = 1

        while (not self.env_is_terminal_fct(state)) and (time_step < self.horizon):
            # env step with random action
            action = self.possible_action_fct.sample()
            next_state = self.env_transition_fct(state, action)
            reward = self.env_reward_fct(state, action, next_state)

            # incrementally computes return
            ret = ret + powered_discount_factor * reward

            powered_discount_factor = powered_discount_factor * self.discount_factor
            state = next_state
            time_step += 1
        return ret


class HeuristicDefaultPolicy(object):
    """
    """

    def __init__(self, heuristic_value_fct, discount_factor, scaling=0.5):
        self.heuristic_value_fct = heuristic_value_fct
        self.discount_factor = discount_factor
        self.scaling = scaling

    def __call__(self, node):
        state = node.state
        # scaling is used to diminish value computed from heuristic because it is a overestimation
        # cf. if some actions are not available in the future (cf archi messages not causing the right action)
        return self.scaling * self.heuristic_value_fct(state, self.discount_factor)
