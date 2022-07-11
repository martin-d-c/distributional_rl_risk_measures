import numpy as np

import torch
import torch.optim as optim
from torch.distributions import Categorical

from tqdm.notebook import trange

from portfolio.portfolio_class import Portfolio
from src.network import PolicyNetPortfolio

def CMV_actor_critic(policy_model:PolicyNetPortfolio, portfolio:Portfolio, n_epochs:int, init_state:int=0,
                        lambd:float=0.5, lr:float=0.001, alpha:float=0.3):
    """
    ## Description
    Optimizes policy_model with respect to chaotic mean-variance risk measure using actor-critic algorithm
    ## Parameters
    policy_model (PolicyNetPortfolio) : neural network to optimize
    portfolio (Portfolio) : considered portfolio
    n_epochs (int) : number of episodes to sample
    init_state (int) : initiale state of each episode
    lambd (float) : penalization factor
    lr (float) : learning rate
    alpha (float) : step of optimization
    ## Returns
    Array of risk measure values during training
    """
    optimizer = optim.Adam(policy_model.parameters(), lr=lr)
    tab_risk_measure = np.zeros(n_epochs)

    v = torch.zeros((portfolio.n_states,portfolio.T)) # Estimation of the expectation of the returns 
    u = torch.zeros((portfolio.n_states,portfolio.T)) # Estimation of the expectation of the quadratic variation of the chaotic returns

    N = np.zeros((portfolio.n_states, portfolio.num_actions)) #Number of visits to each state-action pair
    R = np.zeros((portfolio.n_states, portfolio.num_actions)) #Estimated mean reward for each state-action pair

    for epoch in trange(n_epochs):
        portfolio.reset(init_state)
        for t in range(portfolio.T-1):            
            state = portfolio.state

            # choose an action
            probs = policy_model.forward(torch.tensor(state, dtype=torch.float).unsqueeze(0))
            sampler = Categorical(probs)
            action = sampler.sample()

            # use that action in the environment
            new_state, reward = portfolio.act(action)

            # update N and R
            N[state,action]+=1
            R[state,action] += (reward-R[state,action])/N[state,action]

            # TD erros
            delta_1 = reward + v[new_state,t+1] - v[state,t]
            delta_2 = (reward-R[state,action])**2 + u[new_state,t+1] - u[state,t]

            # update v and u
            v[state,t] = v[state,t] + alpha*delta_1
            u[state,t] = u[state,t] + alpha*delta_2

            # calculate loss    
            log_prob = -sampler.log_prob(action)  
            loss = (delta_1-lambd*delta_2)*log_prob-sampler.entropy()

            # Train the policy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tab_risk_measure[epoch] = v[init_state, 0] - lambd*u[init_state,0]
    policy_model.save("policy_parameters_actor_critic_portfolio")
    return tab_risk_measure

def CMV_actor_critic_dis(policy_model:PolicyNetPortfolio, portfolio:Portfolio, n_epochs:int, m:int, V_min:float, 
                            V_max:float, init_state:int=0, lambd:float=0.5, lr:float=0.001, alpha:float=0.3):
    """
    ## Description
    Optimizes policy_model with respect to chaotic mean-variance risk measure using actor-critic algorithm
    and distributional reinforcement learning
    ## Parameters
    policy_model (PolicyNetPortfolio) : neural network to optimize
    portfolio (Portfolio) : considered portfolio
    n_epochs (int) : number of episodes to sample
    m (int) : number of points of the representation
    V_min, V_max (float) : minimal and maximal returns
    init_state (int) : initiale state of each episode
    lambd (float) : penalization factor
    lr (float) : learning rate
    alpha (float) : step of optimization
    ## Returns
    Array of risk measure values during training
    """
    optimizer = optim.Adam(policy_model.parameters(), lr=lr)
    tab_risk_measure = np.zeros(n_epochs)

    # Estimation of the distribution of the returns
    P_v = np.random.rand(m, portfolio.n_states, portfolio.T, portfolio.num_actions)
    P_v[1:,:,-1,:] = np.zeros((m-1,portfolio.n_states,portfolio.num_actions))    
    P_v /= P_v.sum(axis=0)

    # Estimation of the distribution of the quadratic variation of the chaotic returns
    P_u = np.random.rand(m, portfolio.n_states, portfolio.T, portfolio.num_actions)
    P_u[1:,:,-1,:] = np.zeros((m-1,portfolio.n_states,portfolio.num_actions))    
    P_u /= P_u.sum(axis=0)

    theta = np.linspace(V_min, V_max, m)


    N = np.zeros((portfolio.n_states, portfolio.num_actions)) #Number of visits to each state-action pair
    R = np.zeros((portfolio.n_states, portfolio.num_actions)) #Estimated mean reward for each state-action pair

    for epoch in trange(n_epochs):
        portfolio.reset(init_state)
        for t in range(portfolio.T-1):            
            state = portfolio.state

            # choose an action
            probs = policy_model.forward(torch.tensor(state, dtype=torch.float).unsqueeze(0))
            sampler = Categorical(probs)
            action = sampler.sample()

            # use that action in the environment
            new_state, reward = portfolio.act(action)

            # update N and R
            N[state,action]+=1
            R[state,action] += (reward-R[state,action])/N[state,action]

            # TD erros
            q_action_v = np.argmax([np.sum(P_v[:, new_state,t+1,a]*theta) for a in range(portfolio.num_actions)])
            q_action_u = np.argmax([np.sum(P_u[:, new_state,t+1,a]*theta) for a in range(portfolio.num_actions)])

            delta_1 = reward + np.sum(P_v[:, new_state,t+1,q_action_v]*theta) - np.sum(P_u[:, state,1,action]*theta)
            delta_2 = (reward-R[state,action])**2 + np.sum(P_v[:, new_state,t+1,q_action_u]*theta) - np.sum(P_u[:, new_state,t,action]*theta)

            # updating P_v and P_u

            P_update = np.zeros(m)
            for j in range(m):
                g = reward + theta[j]
                if g <= theta[0]:
                    P_update[0] += P_v[j,new_state,t+1,q_action_v]
                elif g >= theta[-1]:
                    P_update[-1] += P_v[j,new_state,t+1,q_action_v]
                else:
                    i_star = np.where(theta <=g)[0][-1]
                    zeta = (g-theta[i_star])/(theta[i_star+1]-theta[i_star])
                    P_update[i_star] += P_v[j,new_state,t+1,q_action_v]*(1-zeta)
                    P_update[i_star+1] += P_v[j,new_state,t+1,q_action_v]*zeta 
            P_v[:,state,t,action] = (1-alpha)*P_v[:,state,t,action] + alpha*P_update

            P_update = np.zeros(m)
            for j in range(m):
                g = (reward-R[state,action])**2 + theta[j]
                if g <= theta[0]:
                    P_update[0] += P_u[j,new_state,t+1,q_action_u]
                elif g >= theta[-1]:
                    P_update[-1] += P_u[j,new_state,t+1,q_action_u]
                else:
                    i_star = np.where(theta <=g)[0][-1]
                    zeta = (g-theta[i_star])/(theta[i_star+1]-theta[i_star])
                    P_update[i_star] += P_u[j,new_state,t+1,q_action_u]*(1-zeta)
                    P_update[i_star+1] += P_u[j,new_state,t+1,q_action_u]*zeta 
            P_u[:,state,t,action] = (1-alpha)*P_u[:,state,t,action] + alpha*P_update

            # calculate loss    
            log_prob = -sampler.log_prob(action)  
            loss = (delta_1-lambd*delta_2)*log_prob-sampler.entropy()

            # Train the policy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tab_risk_measure[epoch] = np.max([np.sum((P_v[:, init_state,0,a]-lambd*P_u[:, init_state,0,a])*theta) for a in range(portfolio.num_actions)])
    policy_model.save("policy_parameters_actor_critic_portfolio")
    return tab_risk_measure