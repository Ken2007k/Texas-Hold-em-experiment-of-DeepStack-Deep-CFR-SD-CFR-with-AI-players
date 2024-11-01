
---Project Structure---

This project accomplishes a battle among six AI players using three algorithms—DeepStack, DeepCFR, and SDCFR—in a nine-player No-Limit Texas Hold'em setting. The AI players are divided into three difficulty levels: beginner (random strategy), intermediate (basic CFR strategy), and advanced (CFRPlus strategy, implemented using PokerRL's built-in functionality). The project files include source code (src), saved model weights (trained_agents, containing only the final experiment since previous ones were accidentally overwritten), three reference papers, loss curve graphs for each model during training, and a results table showing mbb/g evaluations for the three models.

The source code files contain English comments and are mainly based on the GitHub project Deep-CFR, with modifications to adjust the game type and dependency files. The DeepStack implementation is adapted from PyStack and rewritten using the torch framework to integrate it within the Deep-CFR environment.


---Results Analysis---

From the loss curve, we can see that SD-CFR and Deep-CFR follow a similar trend, although SDCFR converges faster. DeepStack shows more fluctuations as it converges.

The result is quite interesting: DeepStack is a "strengthens with stronger opponents" algorithm, achieving a significant payoff against intermediate and advanced players. This may be due to the CFRresolve step at the end of the algorithm process, which makes DeepStack overly cautious against random opponents, limiting its ability to maximize profit. However, as the opponent's strategy becomes more rational, DeepStack’s advantage gradually becomes evident. SDCFR not only improves in convergence speed but also surpasses DeepCFR in final payoff. In addition, because of the high convergence speed, SD-CFR can easily study the simple and random strategies of beginner, so it has the best performance playing against beginner-level players.
