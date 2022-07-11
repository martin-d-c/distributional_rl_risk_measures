import numpy as np
import datetime, random

from maze.maze_class import Qmaze
from src.utils import format_time

def qtrain_mv_chaotic(qmaze:Qmaze, n_epochs:int, lambd:float=0.5, alpha=0.3, num_actions:int=4, **opt):
    """
    ## Description
    Q-Learning optimizing chaotic mean-variance risk measure 
    ## Parameters
    qmaze (Qmaze) : considered maze
    n_epochs (int) : number of episodes to sample
    lambd (float): penalization factor
    alpha (float) : learning rate
    num_actions (int) : size of the action space
    ## Returns
    Array of risk measure values during training
    Estimated state-action value function
    """

    start_time = datetime.datetime.now()
    tab_risk_measure = np.zeros(n_epochs)

    nrows, ncols = qmaze._maze.shape

    eps_start = opt.get('eps_start',0.9) 
    eps_end = opt.get('eps_end',0.05)
    eps_decay = opt.get('eps_decay',0.99)
    steps_done = 0

    Q = np.zeros((nrows, ncols, num_actions)) #Estimated Q-values
    N = np.zeros((nrows, ncols, num_actions)) #Number of visits to each state-action pair
    R = np.zeros((nrows, ncols, num_actions)) #Estimated mean reward for each state-action pair


    for epoch in range(n_epochs):
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())

        rat_cell = random.choice(qmaze.free_cells)
        qmaze.reset(rat_cell)
        game_over = False
        
        while not game_over:
            # Choose an action according to epsilon-greedy policy
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_row, prev_col,_ = qmaze.state
            eps_threshold = eps_end + (eps_start - eps_end) * eps_decay ** steps_done
            steps_done += 1
            if np.random.rand() < eps_threshold:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(Q[prev_row,prev_col,:])
            
            # use that action in the environment
            _, reward, status = qmaze.act(action)
            row, col,_ = qmaze.state

            if status == 'win' or status == 'lose':
                game_over = True

            # Update the estimated Q-values
            N[prev_row,prev_col,action] += 1
            R[prev_row,prev_col,action] += (reward-R[prev_row,prev_col,action])/N[prev_row,prev_col,action]
            Q[prev_row,prev_col,action] = (1-alpha)*Q[prev_row,prev_col,action] + alpha*(reward-lambd*(reward-R[prev_row,prev_col,action])**2 + np.max(Q[row,col,:]))
        
        tab_risk_measure[epoch-1] = np.max(Q[0,0,:])
        template = "Epoch: {:03d}/{:d} | Risk Measure: {:.4f} | time: {}"
        print(template.format(epoch, n_epochs, tab_risk_measure[epoch-1], t))
    
    return tab_risk_measure, Q

def qtrain_mv_chaotic_dis(m:int, V_min:float, V_max:float, qmaze:Qmaze, n_epochs:int, lambd:float=0.5, 
             alpha=0.3, num_actions:int=4, **opt):
    """
    ## Description
    Distributional Q-Learning optimizing chaotic mean-variance risk measure 
    ## Parameters
    m (int) : number of points of the representation
    V_min, V_max (float) : minimal and maximal returns 
    qmaze (Qmaze) : considered maze
    n_epochs (int) : number of episodes to sample
    lambd (float): penalization factor
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

    N = np.zeros((nrows, ncols, num_actions)) #Number of visits to each state-action pair
    R = np.zeros((nrows, ncols, num_actions)) #Estimated mean reward for each state-action pair

    for epoch in range(n_epochs):
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())

        rat_cell = random.choice(qmaze.free_cells)
        qmaze.reset(rat_cell)
        game_over = False

        while not game_over:
            # Choose an action according to epsilon-greedy policy
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_row, prev_col,_ = qmaze.state
            eps_threshold = eps_end + (eps_start - eps_end) * eps_decay ** steps_done
            steps_done += 1
            if np.random.rand() < eps_threshold:
                action = random.choice(valid_actions)
            else:
                action = np.argmax([np.sum(P[:, prev_row, prev_col,a]*theta) for a in range(num_actions)])
            
            # use that action in the environment
            _, reward, status = qmaze.act(action)
            row, col,_ = qmaze.state
            if status == 'win' or status == 'lose':
                game_over = True

            # Categorical Q-learning
            q_action = np.argmax([np.sum(P[:, row, col,a]*theta) for a in range(num_actions)])
            N[prev_row,prev_col,action] += 1
            R[prev_row,prev_col,action] += (reward-R[prev_row,prev_col,action])/N[prev_row,prev_col,action]
            P_update = np.zeros(m)
            for j in range(m):
                g = reward - lambd*(reward-R[prev_row,prev_col,action])**2 + theta[j]
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

       
        tab_risk_measure[epoch-1] = np.max([np.sum(P[:, 0, 0,a]*theta) for a in range(num_actions)])
        template = "Epoch: {:03d}/{:d} | Risk Measure: {:.4f} | time: {}"
        print(template.format(epoch, n_epochs, tab_risk_measure[epoch-1], t))
    
    return tab_risk_measure, P

def qtrain_mv_chaotic_dis_double(m:int, V_min:float, V_max:float, qmaze:Qmaze, n_epochs:int, lambd:float=0.5, 
             alpha=0.3, num_actions:int=4, **opt):
    """
    ## Description
    Distributional Q-Learning optimizing chaotic mean-variance risk measure with double Q-learning
    (Learning distribution of returns and distribution of penalized returns)    ## Parameters
    m (int) : number of points of the representation
    V_min, V_max (float) : minimal and maximal returns 
    qmaze (Qmaze) : considered maze
    n_epochs (int) : number of episodes to sample
    lambd (float): penalization factor
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

    P_pen = np.random.rand(m, nrows, ncols, num_actions)
    P_pen /= P_pen.sum(axis=0)

    P_ret = np.random.rand(m, nrows, ncols, num_actions)
    P_ret /= P_ret.sum(axis=0)


    for epoch in range(n_epochs):
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())

        rat_cell = random.choice(qmaze.free_cells)
        qmaze.reset(rat_cell)
        game_over = False

        while not game_over:
            # Choose an action according to epsilon-greedy policy
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_row, prev_col,_ = qmaze.state
            eps_threshold = eps_end + (eps_start - eps_end) * eps_decay ** steps_done
            steps_done += 1
            if np.random.rand() < eps_threshold:
                action = random.choice(valid_actions)
            else:
                action = np.argmax([np.sum(P_pen[:, prev_row, prev_col,a]*theta) for a in range(num_actions)])
            
            # use that action in the environment
            _, reward, status = qmaze.act(action)
            row, col,_ = qmaze.state
            if status == 'win' or status == 'lose':
                game_over = True

            # Categorical Q-learning
            q_action = np.argmax([np.sum(P_pen[:, row, col,a]*theta) for a in range(num_actions)])
            # For returns
            P_ret_update = np.zeros(m)
            for j in range(m):
                g = reward + theta[j]
                if g <= theta[0]:
                    P_ret_update[0] += P_ret[j,row,col,q_action]
                elif g >= theta[-1]:
                    P_ret_update[-1] += P_ret[j,row,col,q_action]
                else:
                    i_star = np.where(theta <=g)[0][-1]
                    zeta = (g-theta[i_star])/(theta[i_star+1]-theta[i_star])
                    P_ret_update[i_star] += P_ret[j,row,col,q_action]*(1-zeta)
                    P_ret_update[i_star+1] += P_ret[j,row,col,q_action]*zeta 
            P_ret[:,prev_row,prev_col,action] = (1-alpha)*P_ret[:,prev_row,prev_col,action] + alpha*P_ret_update
            #For penalized returns
            P_pen_update = np.zeros(m)
            for j in range(m):
                g = reward - lambd*(reward-np.sum(P_ret[:,prev_row,prev_col,action]*theta))**2 + theta[j]
                if g <= theta[0]:
                    P_pen_update[0] += P_pen[j,row,col,q_action]
                elif g >= theta[-1]:
                    P_pen_update[-1] += P_pen[j,row,col,q_action]
                else:
                    i_star = np.where(theta <=g)[0][-1]
                    zeta = (g-theta[i_star])/(theta[i_star+1]-theta[i_star])
                    P_pen_update[i_star] += P_pen[j,row,col,q_action]*(1-zeta)
                    P_pen_update[i_star+1] += P_pen[j,row,col,q_action]*zeta 
            P_pen[:,prev_row,prev_col,action] = (1-alpha)*P_pen[:,prev_row,prev_col,action] + alpha*P_pen_update

       
        tab_risk_measure[epoch-1] = np.max([np.sum(P_pen[:, 0, 0,a]*theta) for a in range(num_actions)])
        template = "Epoch: {:03d}/{:d} | Risk Measure: {:.4f} | time: {}"
        print(template.format(epoch, n_epochs, tab_risk_measure[epoch-1], t))
    
    return tab_risk_measure, P_pen, P_ret