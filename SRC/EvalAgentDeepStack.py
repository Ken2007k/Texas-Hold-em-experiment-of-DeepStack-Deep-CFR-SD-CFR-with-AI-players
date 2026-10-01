import numpy as np
import DeepStack.NnTraining.train as DS

class EvalAgentDeepStack:
    """ DeepStack-based Evaluation Agent. Utilizes DeepStack methods like belief updating, CFR with deep learning. """
    DeepStack = "DeepStack"
    def __init__(self, n_actions, neural_net, initial_belief=None):
        self.n_actions = n_actions
        self.neural_net = neural_net  # Deep neural network representing the strategy
        self.belief = initial_belief if initial_belief is not None else np.ones(n_actions) / n_actions
        self.strategy = np.ones(n_actions) / n_actions  # Uniform initial strategy
        self.iteration = 0  # Track number of iterations
        self.strategy_sum = np.zeros(n_actions)  # Cumulative strategy for averaging

    def reset(self):
        """ Resets the agent's internal state, including belief and strategy. """
        self.belief.fill(1.0 / self.n_actions)
        self.strategy.fill(1.0 / self.n_actions)
        self.strategy_sum.fill(0)
        self.iteration = 0

    def evaluate_strategy(self, game_state):
        """
        Evaluate the current strategy in a given game state using DeepStack’s evaluate function.
        The evaluation incorporates game tree traversal and neural net predictions.
        """
        evaluation = DS.evaluate_strategy(self.strategy, game_state, self.neural_net)
        return evaluation

    def update_belief(self, observed_action):
        """
        Update the belief about the opponent’s strategy using DeepStack’s belief update mechanism.
        This function assumes the observed action from the opponent.
        """
        self.belief = DS.update_belief(self.belief, observed_action)

    def select_action(self, game_state):
        """
        Selects an action based on the current strategy. The strategy is derived from the neural network.
        Simulates a game and updates beliefs using DeepStack mechanisms.
        """
        strategy = self.get_strategy(game_state)
        action = np.random.choice(np.arange(self.n_actions), p=strategy)
        self.update_strategy_sum(strategy)  # Update the cumulative strategy
        return action

    def get_strategy(self, game_state):
        """
        Retrieve the current strategy using DeepStack’s neural network-based CFR update.
        DeepStack uses deep neural networks to generate strategies based on game states.
        """
        self.strategy = DS.deep_cfr_update(self.strategy, game_state, self.neural_net)
        return self.strategy

    def update_strategy_sum(self, strategy):
        """ Updates the cumulative strategy sum (used for averaging strategies over time). """
        self.strategy_sum += strategy
        self.iteration += 1

    def get_average_strategy(self):
        """ Returns the average strategy accumulated over all iterations. """
        if self.iteration > 0:
            return self.strategy_sum / self.iteration
        return np.ones(self.n_actions) / self.n_actions  # Uniform if no iterations






# Example usage:
if __name__ == "__main__":
    # Assume a neural network model (can be a placeholder for simplicity here)
    neural_net = "deepstack_neural_net_placeholder"

    agent = EvalAgentDeepStack(n_actions=5, neural_net=neural_net)
    agent.reset()

    # Simulate a few iterations
    for _ in range(10):
        game_state = np.random.randn(10)  # Simulate some game state
        action = agent.select_action(game_state)
        print(f"Selected action: {action}")

        # Simulate an opponent's action and update belief
        observed_action = np.random.choice(5)
        agent.update_belief(observed_action)

        # Simulate a game outcome
        result = agent.simulate_game(game_state)
        print(f"Simulated game result: {result}")

    # Final strategy and average strategy after simulation
    print("Final strategy:", agent.get_strategy(game_state))
    print("Average strategy:", agent.get_average_strategy())
