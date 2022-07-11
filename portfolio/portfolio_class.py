import numpy as np
import numpy.random as npr

LV = 0
MV = 1
HV = 2
states_dict = {
    LV : "Low-Vol",
    MV : "Medium-Vol",
    HV : "High-Vol",
}

mu_ = [0.2,0.6,1.] # risk free rates
sigma_ = [0.5,1.,1.5] #volatilities

class Portfolio(object):
    def __init__(self, init_state:int=0, T:int=20,  max_invest:int=5, mu:list=mu_, sigma:list=sigma_):
        """
        ## Description
        Portfolio with three states : Low, Medium and High volatility
        ## Arguments 
        init_state (int) : Initial state
        T (int) : Number of timesteps
        max_invest (int) : Maximum total investment (risk-free + risky asset)
        mu (list) : Risk free rate for each state
        sigma (list) : Volatility for each state
        """

        self.mu = mu
        self.sigma = sigma
        self.T = T
        actions = []
        for i in range(max_invest+1):
            for j in range(max_invest+1-i):
                actions.append((i,j))
        self.max_invest = max_invest
        self.actions = actions
        self.num_actions = len(actions)
        self.reset(init_state)
        self.n_states = 3
        
        # transition matrix : P[i,j] = prob(s'=j|q_R=i)
        P = [
                [ 0.5,0.45,0.05 ],
                [ 1/3,1/3,1/3 ],
                [ 1/3,1/3,1/3 ],
                [ 0.1,0.45,0.45 ],
                [ 0.1,0.45,0.45 ],
                [ 0.05,0.25,0.7 ]
            ]
        P = np.array(P)
        self.P = P

    def reset(self, init_state:int):
        self.state = init_state
        self.total_reward = 0


    def act(self, idx_action:int):
        """
        ## Description
        Act with the idx_action-th action
        ## Returns
        A tuple : (new state, reward)
        """
        action = self.actions[idx_action]
        q_RF,q_R = action

        #calculate reward
        reward = q_RF*self.mu[self.state] + q_R*(self.mu[self.state] +self.sigma[self.state]*npr.randn() )
        self.total_reward += reward

        #state update
        prob = self.P[q_R,:]
        self.state = npr.choice([0,1,2],p=prob)
        return self.state, reward