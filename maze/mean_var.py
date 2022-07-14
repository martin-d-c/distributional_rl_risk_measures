import numpy as np
import datetime, random
import matplotlib.pyplot as plt
from tqdm.notebook import trange

import torch
import torch.optim as optim
from torch.distributions import Categorical

from src.network import PolicyNet
from src.utils import format_time
from maze.maze_class import Qmaze

def qtrain_mv(m:int, V_min:float, V_max:float, qmaze:Qmaze, n_epochs:int, lambd:float=0.5, 
            gamma:float=1., alpha=0.3, num_actions:int=4, **opt):
    """
    ## Description
    Distributional Q-Learning optimizing mean-variance risk measure 
    ## Parameters
    m (int) : number of points of the representation
    V_min, V_max (float) : minimal and maximal returns 
    qmaze (Qmaze) : considered maze
    n_epochs (int) : number of episodes to sample
    lambd (float): penalization factor
    gamma (float) : discount factor
    alpha (float) : learning rate
    num_actions (int) : size of the action space
    ## Returns
    Array of risk measure values during training
    Weights of the categorical representation for each cell/action
    """

    start_time = datetime.datetime.now()
    tab_risk_measure = np.zeros(n_epochs)

    theta = np.linspace(V_min, V_max, m)
    nrows, ncols = qmaze._maze.shape

    eps_start = opt.get('eps_start',0.9) 
    eps_end = opt.get('eps_end',0.05)
    eps_decay = opt.get('eps_decay',0.99)
    steps_done = 0

    P = np.random.rand(m, nrows, ncols, num_actions)
    P /= P.sum(axis=0)

    for epoch in range(n_epochs):
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())

        rat_cell = random.choice(qmaze.free_cells)
        qmaze.reset(rat_cell)
        game_over = False

        while not game_over:
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_row, prev_col,_ = qmaze.state
            eps_threshold = eps_end + (eps_start - eps_end) * eps_decay ** steps_done
            steps_done += 1

            # Choose an action according to epsilon-greedy policy
            if np.random.rand() < eps_threshold:
                action = random.choice(valid_actions)
            else:
                action = np.argmax([np.sum(P[:, prev_row, prev_col,a]*theta) -
                        lambd*(np.sum(P[:, prev_row, prev_col,a]*theta**2) - np.sum(P[:, prev_row, prev_col,a]*theta)**2 )
                        for a in range(num_actions)])
            
            _, reward, status = qmaze.act(action)
            row, col,_ = qmaze.state

            if status == 'win' or status == 'lose':
                game_over = True

            q_action = np.argmax([np.sum(P[:, row, col,a]*theta) -
                        lambd*(np.sum(P[:, row, col,a]*theta**2)-np.sum(P[:, row, col,a]*theta)**2) 
                        for a in range(num_actions)])

            # Categorical Q-learning
            P_update = np.zeros(m)
            for j in range(m):
                g = reward + gamma * theta[j]
                if g <= theta[0]:
                    P_update[0] += P[j,row,col,q_action]
                elif g >= theta[-1]:
                    P_update[-1] += P[j,row,col,q_action]
                else:
                    i_star = np.where(theta <=g)[0][-1]
                    zeta = (g-theta[i_star])/(theta[i_star+1]-theta[i_star])
                    P_update[i_star] += P[j,row,col,q_action]*(1-zeta)
                    P_update[i_star+1] += P[j,row,col,q_action]*zeta 
            P[:,prev_row,prev_col,action] = (1-alpha)*P[:,prev_row,prev_col,action] + alpha*P_update

        tab_risk_measure[epoch-1] = np.max([np.sum(P[:, 0, 0,a]*theta) -
                        lambd*(np.sum(P[:, row, col,a]*theta**2)-np.sum(P[:, row, col,a]*theta)**2) 
                        for a in range(num_actions)])
        template = "Epoch: {:03d}/{:d} | Risk Measure: {:.4f} | time: {}"
        print(template.format(epoch, n_epochs, tab_risk_measure[epoch-1], t))
    
    return tab_risk_measure, P

def mean_var_training_actor_critic_fenchel_reg(policy_model:PolicyNet, qmaze:Qmaze, n_epoch:int,
    lambd:float=0.5, gamma:float=1., lr:float=0.001, alpha:float=0.3, alpha_y:float=0.3, tau:float=0.1):
    """
    ## Description
    Optimizes policy_model with respect to mean-variance risk measure using actor-critic algorithm,
    Legendre-Fenchel duality and entropic regularization
    ## Parameters
    policy_model (PolicyNet) : neural network to optimize
    qmaze (Qmaze) : considered maze
    n_epoch (int) : number of episodes to sample
    lambd (float): penalization factor
    gamma (float) : discount factor
    alpha (float) : learning rate for value functions
    alpha_y (float) : learning rate for y
    num_actions (int) : size of the action space
    tau (float) : intensity of entropic regularization
    ## Returns
    Array of risk measure values during training
    """
    optimizer = optim.Adam(policy_model.parameters(), lr=lr)

    start_time = datetime.datetime.now()
    tab_risk_measure = np.zeros(n_epoch)

    nrows, ncols = qmaze._maze.shape
    v_1 = torch.zeros((nrows, ncols)) # estimate of the value function
    v_2 = torch.zeros((nrows, ncols)) # estimate of the expectation of the square of the returns

    y = torch.tensor(0., dtype=torch.float) # for Legendre-Fenchel duality
    for epoch in range(1,n_epoch+1):

        #Generate an episode
        rat_cell = random.choice(qmaze.free_cells)
        qmaze.reset((4,2))
            
        qmaze.reset(rat_cell)
        state = qmaze.observe()
        I=1
        while True:
            # calculate probabilities of taking each action
            probs = policy_model.forward(torch.tensor(state[0], dtype=torch.float).unsqueeze(0))
            # sample an action from that set of probs
            sampler = Categorical(probs)
            action = sampler.sample()

            # use that action in the environment
            prev_row, prev_col,_ = qmaze.state
            new_state, reward, status = qmaze.act(action)
            row, col,_ = qmaze.state


            delta_1 = reward + gamma*v_1[row,col] - v_1[prev_row,prev_col]
            delta_2 = reward**2 +2*gamma*reward*v_1[row,col] + gamma**2*v_2[row,col] - v_2[prev_row,prev_col]

            # updating moment functions
            v_1[prev_row, prev_col] = v_1[prev_row, prev_col] + alpha*delta_1
            v_2[prev_row, prev_col] = v_2[prev_row, prev_col] + alpha*delta_2

            # Calculate log-probability of the action taken
            #probs = policy_model(torch.cat([states, torch.arange(states.shape[0]).unsqueeze(1), G.unsqueeze(1)], dim=1))
            probs = policy_model.forward(torch.tensor(state[0], dtype=torch.float).unsqueeze(0))
            sampler = Categorical(probs)
            log_prob = -sampler.log_prob(action)  

            #updating y
            y+= alpha_y*(2*(reward+v_1[row, col]) + 1/lambd - 2*y)

            # calculate loss_1  
            loss_1 = log_prob * delta_1 

            # calculate loss_2
            if status == 'win' or status == 'lose':
                loss_2 = delta_2*log_prob
            else :
                probs = policy_model.forward(torch.tensor(new_state[0], dtype=torch.float).unsqueeze(0))
                # sample an action from that set of probs
                sampler = Categorical(probs)
                action_2 = sampler.sample()
                log_prob_2 = -sampler.log_prob(action_2)
                row_2, col_2, reward_2 = qmaze.new_state_reward(action)
                delta_1_2 = reward_2 + gamma*v_1[row_2,col_2] - v_1[row,col]
                loss_2 = delta_2*log_prob + 2*reward*delta_1_2*log_prob_2
            
            loss = I*(2*y*loss_1-loss_2) - tau*sampler.entropy()

            #Train the policy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = new_state
            I = gamma*I
            if status == 'win' or status == 'lose':
                break
            
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        tab_risk_measure[epoch-1] = v_1[0, 0] - lambd*(v_2[0, 0]-v_1[0,0]**2)
        template = "Epoch: {:03d}/{:d} | Risk Measure: {:.4f} | time: {}"
        print(template.format(epoch, n_epoch, tab_risk_measure[epoch-1], t))
    
    policy_model.save("policy_parameters_actor_critic_fenchel")
    return tab_risk_measure       