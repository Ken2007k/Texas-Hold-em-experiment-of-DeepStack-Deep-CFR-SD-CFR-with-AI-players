import numpy as np

class EvalAgentBasicCFR:
    RANDOM = "RANDOM"
    CFR = "CFR"
    CFR_PLUS = "CFR_PLUS"
    ALL_MODES = [RANDOM,CFR, CFR_PLUS]

    def __init__(self, n_actions, mode=None):
        self.n_actions = n_actions
        self.regrets = np.zeros(n_actions)  # Regret for each action
        self.strategy = np.ones(n_actions) / n_actions  # Uniform initial strategy
        self.strategy_sum = np.zeros(n_actions)  # Cumulative strategy for averaging
        self.regret_sum = np.zeros(n_actions)  # Cumulative regrets
        self.mode = mode if mode in self.ALL_MODES else self.CFR
        self.iteration = 0  # To track which iteration we are in

    def reset(self):
        """ Resets the agent's internal state (regrets and strategy). """
        self.regrets.fill(0)
        self.strategy.fill(1.0 / self.n_actions)
        self.strategy_sum.fill(0)
        self.regret_sum.fill(0)
        self.iteration = 0

    def set_mode(self, mode):
        """ Changes the evaluation mode (CFR, linear CFR, or CFR+). """
        if mode in self.ALL_MODES:
            self.mode = mode
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def notify_of_reset(self):
        """ Notifies the agent that the environment is resetting. Resets the agent. """
        self.reset()

    def get_strategy(self):
        """ Returns the current strategy based on regrets. """
        if self.mode == self.CFR or self.mode == self.LINEAR_CFR:
            # Normal CFR: Positive regrets form the strategy
            positive_regrets = np.maximum(self.regrets, 0)
            regret_sum = np.sum(positive_regrets)
            if regret_sum > 0:
                self.strategy = positive_regrets / regret_sum
            else:
                self.strategy.fill(1.0 / self.n_actions)  # Uniform strategy if no positive regrets

        elif self.mode == self.CFR_PLUS:
            # CFR+: Use only positive regrets, don't subtract past regrets
            positive_regrets = np.maximum(self.regret_sum, 0)
            regret_sum = np.sum(positive_regrets)
            if regret_sum > 0:
                self.strategy = positive_regrets / regret_sum
            else:
                self.strategy.fill(1.0 / self.n_actions)

        return self.strategy

    def select_action(self):
        """ Selects an action based on the current strategy. """
        strategy = self.get_strategy()
        action = np.random.choice(np.arange(self.n_actions), p=strategy)
        self.update_strategy_sum(strategy)  # Track cumulative strategy for averaging
        return action

    def update_regrets(self, action, regret):
        """ Updates the regrets based on the outcome of the action taken. """
        self.regrets[action] += regret
        if self.mode == self.CFR_PLUS:
            self.regret_sum[action] += max(0, regret)  # CFR+ only accumulates positive regrets

    def update_strategy_sum(self, strategy):
        """ Updates the cumulative strategy sum (used for linear CFR and averaging). """
        self.strategy_sum += strategy
        self.iteration += 1

    def get_average_strategy(self):
        """ Returns the average strategy across iterations (used in linear CFR). """
        if self.iteration > 0:
            return self.strategy_sum / self.iteration
        return np.ones(self.n_actions) / self.n_actions  # Uniform if no iterations

