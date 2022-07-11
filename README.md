# **Distributional Reinforcement Learning and Risk Measures**

This repository regroups the main methods implemented during my internship at [UQAM](https://sciences.uqam.ca) in summer 2022 about *Distributional Reinforcement Learning and Risk Measures* supervised  by Arthur Charpentier.

This project includes two differents environments described below and regroups several implementation of reinforcement learning and distributional reinforcement learning methods with a focus on risk-sensitive optimization.


## **1.Project Structure**


```
.
├── README.md
├── demo : various notebooks using implemented methods
    ├── change_env.ipynb : effect of change of environment during training
    ├── maze_training.ipynb :  deep Q-learning and distributional reinforcement learning evaluation 
    ├── maze_risk.ipynb : policies optimization according to different risk-related criteria
    ├── portfolio_optimization.ipynb : reinforcement learning in the portfolio environment
├── maze
    ├── chaotic_mean_var.py : chaotic mean variance optimization
    ├── cvar.py : conditional value at risk optimization
    ├── drl_methods.py : distributional reinforcement learning algorithms
    ├── maze_class.py : define maze environment
    ├── mean_var.py : mean-variance optimization
    ├── rl_methods.py : reinforcement learning algorithms
├── outputs
    ├── change_environment : figures from change_env notebook
├── parameters : store trained parameters
    ├── model_parameters
    ├── policy_parameters
├── portfolio
    ├── portfolio_class.py : define portfolio environment
    ├── portfolio_methods.py : (distributional) reinforcement learning for portfolio optimization
├── src
    ├── network.py
    ├── utils.py

``` 

## **2.Environments description**

### 2.1 Maze

<p align="center">
  <img src="outputs/maze.png" alt="Maze environment" background-color="red" title="Maze environment" width="360" height="200">
</p>

The first environment is a maze made up of 25 cells: a start cell (upper left corner), an arrival cell (lower right corner), 15 free cells (in white), 6 blocked cells (in black) as well as 2 special cells (in red) whose role will depend on the situation. The objective is to teach an agent the optimal path in this maze between the start and the end cell, taking into account the travel time and the effect of the special cells. The possible actions are the four directions and the state of an agent is its row and column indices. Once the action is chosen, the agent stays in its position if the action is invalid (if it takes it to a blocked cell or out of the frame) or performs the chosen movement if the action is valid. The rewards are : $-0.75$ in the case of an invalid action, $-0.04$ if the action is valid but the agent does not reach the target and $1$ if the agent reaches its target.

Two possible strategies are shown in the figure: the risky strategy is the optimum in the case where the special cells act as free cells and the risky strategy has a constant state value function regardless of the action of the special cells.

This environment is mainly inspired from [[6]](https://www.samyzaf.com/ML/rl/qmaze.html).




### 2.2 Portfolio

This second environment is directly inspired from the article by Vadori et al. [[5]](https://doi.org/10.1145%2F3383455.3422519). It concerns the problem of investing in two financial assets, one risky and one risk-free. The state space consists of three levels of volatility: *LowVol*, *MediumVol* and *HighVol*. The greater the investment in the risky asset, the greater is the probability of entering a high volatility state. At each time, the investor chooses to invest a quantity $q_R$ in the risky asset and a quantity $q_{RF}$ in the risk-free asset. He then receives a reward $R = q_{RF}\mu(S) + q_R( \mu(S) + \sigma(S)H ) $ with $H$ a standard Gaussian random variable, $\mu(S)$ the risk-free rate in state $S$ (increasing according to the level of volatility of $S$) and $\sigma(S)$ the volatility of state $S$.

The invested quantities $q_R$ and $q_{RF}$ are positive integers satisfying the condition $q_{RF} + q_R \leq q_{max}$ and the episodes are of fixed duration $T$.

## **3.References**
 [1] Marc G. Bellemare, Will Dabney et Mark Rowland. *Distributional Reinforcement
Learning*. http://www.distributional-rl.org. MIT Press, 2022.

 [2] Bo Liu et al. *A Block Coordinate Ascent Algorithm for Mean-Variance Optimization*.2018. url : https://arxiv.org/abs/1809.02292.

 [3] David Silver. *Lectures on Reinforcement Learning*. url : https://www.davidsilver.uk/teaching/. 2015.

 [4] Richard S Sutton et Andrew G Barto. *Reinforcement learning : An introduction*. MIT
press, 2018.

 [5] Nelson Vadori et al. « Risk-sensitive reinforcement learning ». In : *Proceedings of the
First ACM International Conference on AI in Finance*. ACM, 2020. url : https://doi.org/10.1145%2F3383455.3422519.

 [6] Samy Zafrany. *Deep reinforcement learning for maze solving*. url : https://www.samyzaf.com/ML/rl/qmaze.html.



    