import torch
import numpy as np
import scipy.special as sp
from tqdm.notebook import trange

from maze.maze_class import Qmaze
from src.network import Net

def moment_TD_learning(model:Net, m:int, qmaze:Qmaze, n_episode:int, rat_cell:tuple=(0,0), alpha:float=0.3, 
                        gamma:float=1.):
    """
    ## Description
    Extension of TD-Learning to moments
    ## Parameters
    model (Net) : neural network model
    m (int) : number of moments
    qmaze (Qmaze): considered maze
    n_episode (int) : number of episodes to sample
    rat_cell (tuple): position of the rat
    alpha (float) : learning rate
    gamma (float) : discount factor
    ## Returns 
    An estimate of the m first moments of the distribution of returns for each state
    """
    nrows, ncols = qmaze._maze.shape
    M = np.zeros((m+1, nrows, ncols)) # store the moments for each cell
    M[0] = np.ones((nrows, ncols))
    for _ in range(n_episode):
        qmaze.reset(rat_cell)
        envstate = qmaze.observe()
        while True:
            prev_envstate = envstate
            prev_row, prev_col,_ = qmaze.state
            
            # get next action
            q = model.forward(torch.tensor(prev_envstate, dtype=torch.float))
            action = torch.argmax(q[0])

            # apply action, get rewards and new state
            envstate, reward, status = qmaze.act(action)
            row, col,_ = qmaze.state
            if status == 'win':
                M[:,prev_row, prev_col] = reward**np.arange(m+1)
                break
            else:
                for i in range(1,m+1):
                    binom = np.array([sp.binom(i,j) for j in range(i+1)])
                    M[i,prev_row, prev_col] = (1-alpha)*M[i,prev_row, prev_col] + alpha*np.sum( gamma**(i-np.arange(i+1))*binom*reward**np.arange(i+1)*M[i-np.arange(i+1),row, col] )
    return M

def moment_2_TD_learning(model:Net,qmaze:Qmaze, n_episode:int, rat_cell:tuple=(0,0), alpha:float=0.3, 
                        gamma:float=1.):
    """
    ## Description
    Extension of TD-Learning to the two first moments
    ## Parameters
    model (Net) : neural network model
    m (int) : number of moments
    qmaze (Qmaze): considered maze
    n_episode (int) : number of episodes to sample
    rat_cell (tuple): position of the rat
    alpha (float) : learning rate
    gamma (float) : discount factor
    ## Returns 
    An estimate of the 2 first moments of the distribution of returns for each state
    """
    nrows, ncols = qmaze._maze.shape
    m=2
    M = np.zeros((m+1, nrows, ncols))
    M[0] = np.ones((nrows, ncols))
    for _ in range(n_episode):
        qmaze.reset(rat_cell)
        envstate = qmaze.observe()
        while True:
            prev_envstate = envstate
            prev_row, prev_col,_ = qmaze.state
            
            # get next action
            q = model.forward(torch.tensor(prev_envstate, dtype=torch.float))
            action = torch.argmax(q[0])

            # apply action, get rewards and new state
            envstate, reward, status = qmaze.act(action)
            row, col,_ = qmaze.state
            if status == 'win':
                M[:,prev_row, prev_col] = reward**np.arange(m+1)
                break
            else:
                M[1,prev_row, prev_col] = (1-alpha)*M[1,prev_row, prev_col] + alpha*(reward+gamma*M[1,row, col])
                M[2,prev_row, prev_col] = (1-alpha)*M[2,prev_row, prev_col] + alpha*(reward**2+2*gamma*reward*M[1,row, col]+gamma**2*M[2,row, col])
                
    return M
    
def online_categorical_TD_learning(model:Net, m:int, V_min:float, V_max:float, qmaze:Qmaze, n_episode:int,
                                    rat_cell:tuple=(0,0), gamma:float=1., alpha:float=0.3):
    """
    ## Description
    Online categorical TD learning algorithm to learn the distribution of the returns
    ## Parameters
    model (Net) : model to evaluate
    m (int) : number of points of the representation
    V_min, V_max (float) : minimal and maximal returns 
    qmaze (Qmaze) : considered maze
    n_episode (int) : number of episodes to sample
    rat_cell (tuple): position of the rat
    gamma (float) : discount factor
    alpha (float) : learning rate
    ## Returns
    Weights of the categorical representation for each cell
    """
    theta = np.linspace(V_min, V_max, m)
    nrows, ncols = qmaze._maze.shape
    P = np.random.rand(m, nrows, ncols) # weights of the categorical representation for each cell
    P[:,nrows-1, ncols-1] = np.zeros(m)
    P[m-1,nrows-1, ncols-1] = 1
    P /= P.sum(axis=0)
    for _ in trange(n_episode):
        qmaze.reset(rat_cell)
        status = 'not_over'
        envstate = qmaze.observe()
        while status == 'not_over':
            prev_envstate = envstate
            prev_row, prev_col,_ = qmaze.state
            # get next action
            q = model.forward(torch.tensor(prev_envstate, dtype=torch.float))
            action = torch.argmax(q[0])
            envstate, reward, status = qmaze.act(action)
            row, col,_ = qmaze.state
            
            # categorical projection
            P_update = np.zeros(m)
            for j in range(m):
                g = reward + gamma * theta[j]
                if g <= theta[0]:
                    P_update[0] += P[j,row,col]
                elif g >= theta[-1]:
                    P_update[-1] += P[j,row,col]
                else:
                    i_star = np.where(theta <=g)[0][-1]
                    zeta = (g-theta[i_star])/(theta[i_star+1]-theta[i_star])
                    P_update[i_star] += P[j,row,col]*(1-zeta)
                    P_update[i_star+1] += P[j,row,col]*zeta 
            P[:,prev_row,prev_col] = (1-alpha)*P[:,prev_row,prev_col] + alpha*P_update
           
    return P
