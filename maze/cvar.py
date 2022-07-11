import random
import datetime
import numpy as np

from maze.maze_class import Qmaze
from src.utils import format_time

def qtrain_cvar(m:int, V_min:float, V_max:float, qmaze:Qmaze, n_epochs:int, tau=0.9, gamma:float=1., 
                alpha:float=0.3, num_actions:int=4, **opt):
    """
    ## Description
    Distributional Q-Learning optimizing Conditional Value at Risk 
    ## Parameters
    m (int) : number of points of the representation
    V_min, V_max (float) : minimal and maximal returns 
    qmaze (Qmaze) : considered maze
    n_epochs (int) : number of episodes to sample
    tau (float) : CVaR threshold
    lambd (float): penalization factor
    gamma (float) : discount factor
    alpha (float) : learning rate
    num_actions (int) : size of the action space
    ## Returns
    Weights of the categorical representation for each cell/action
    Array of risk measure values during training
    """
    nrows, ncols = qmaze._maze.shape
    P = np.random.rand(m, nrows, ncols,m,num_actions)
    P[:,nrows-1, ncols-1,:,:] = np.zeros((m,m,num_actions))
    P[m-1,nrows-1, ncols-1,:,:] = np.ones((m,num_actions))
    P /= P.sum(axis=0)
    start_time = datetime.datetime.now()
    tab_risk_measure = np.zeros(n_epochs)
    
    theta = np.linspace(V_min, V_max, m)
    tab_B = np.linspace(qmaze.min_reward,1,m)

    eps_start = opt.get('eps_start',0.9) 
    eps_end = opt.get('eps_end',0.05)
    eps_decay = opt.get('eps_decay',0.99)
    steps_done=0

    for epoch in range(n_epochs):
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())

        rat_cell = random.choice(qmaze.free_cells)
        qmaze.reset(rat_cell)
        game_over = False
        prev_row,prev_col,_ = qmaze.state
        actions_init = [np.argmin([np.sum( (tab_B[i_b]-theta)*(tab_B[i_b]-theta >0)*P[:,prev_row,prev_col,i_b,a])
        for a in range(num_actions)]) for i_b in range(m)]
        idx_b = np.argmax([ b -np.sum((b-theta)*(b-theta >0)*P[:,prev_row,prev_col,i_b,actions_init[i_b]])/tau for i_b,b in enumerate(tab_B)])
        
        while not game_over:

            # Choose an action according to epsilon-greedy policy
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_row,prev_col,_ = qmaze.state
            eps_threshold = eps_end + (eps_start - eps_end) * eps_decay ** steps_done
            steps_done+=1
            if np.random.rand() < eps_threshold:
                action = random.choice(valid_actions)
            else:
                action =  np.argmin([ np.sum( (tab_B[idx_b]-theta)*(tab_B[idx_b]-theta >0)*P[:,prev_row,prev_col,idx_b,a])
                for a in range(num_actions) ])
            
            # use that action in the environment
            _, reward, status = qmaze.act(action)
            row,col,_ = qmaze.state
            idx_b = np.argmin(np.abs(tab_B - (tab_B[idx_b]-reward)/gamma))
            if status == 'win' or status == 'lose':
                game_over = True

            # Categorical Q-Learning
            q_action = np.argmin([ np.sum( (tab_B[idx_b]-theta)*(tab_B[idx_b]-theta >0)*P[:,row,col,idx_b,a] )
            for a in range(num_actions)])
            P_update = np.zeros(m)
            for j in range(m):
                g = reward + gamma * theta[j]
                if g <= theta[0]:
                    P_update[0] = P_update[0] + P[:,row,col,idx_b,q_action][j] 
                elif g >= theta[-1]:
                    P_update[-1] = P_update[-1] + P[:,row,col,idx_b,q_action][j]
                else:
                    i_star = np.where(theta <=g)[0][-1]
                    zeta = (g-theta[i_star])/(theta[i_star+1]-theta[i_star])
                    P_update[i_star] = P_update[i_star] + P[:,row,col,idx_b,q_action][j]*(1-zeta)
                    P_update[i_star+1] = P_update[i_star+1] + P[:,row,col,idx_b,q_action][j]*zeta 

            P[:,prev_row,prev_col,idx_b,action] = (1-alpha)*P[:,prev_row,prev_col,idx_b,action] + alpha*P_update
       
        template = "Epoch: {:03d}/{:d} | Risk Measure: {:.4f} | time: {}"
        tab_risk_measure[epoch-1] = np.max([ b -np.sum((b-theta)*(b-theta >0)*P[:,0,0,i_b,a])/tau for i_b,b in enumerate(tab_B) for a in range(num_actions)])
        
        print(template.format(epoch, n_epochs, tab_risk_measure[epoch-1], t))
    
    return P, tab_risk_measure