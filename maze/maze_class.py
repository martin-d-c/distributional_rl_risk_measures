import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from collections import namedtuple, deque


rat_mark = 0.5  # The current rat cell will be painteg by gray 0.5
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

class UnionFind(object):
    """
    ## Description
    Auxiliary class to help generate random mazes
    """
    def __init__(self, n):
        self.id = np.arange(n)
        self.sz = np.ones(n)
    def find(self, p):
        if self.id[p] == p:
            return p
        else:
            return self.find(self.id[p])
    def union(self, p, q):
        i = self.find(p)
        j = self.find(q)
        if i == j:
            return
        if self.sz[i] < self.sz[j]:
            self.id[i] = j
            self.sz[j] += self.sz[i]
        else:
            self.id[j] = i
            self.sz[i] += self.sz[j]
    def connected(self, p, q):
        return self.find(p) == self.find(q)

class Qmaze(object):
    def __init__(self, maze:np.array=np.zeros((5,5)), rat:tuple=(0,0), random_init:bool=True, n_rows:int=5, n_cols:int=5, 
                    p:float=0.6, bombs:int=0, door_mode:bool=False, prob_door_close:float=1., perte:float=0):
        """
        ## Description
        Maze object that keeps track of the state of the maze, rat location
        Two possible ways of initializing the maze: 
        1) provide a maze matrix
        2) generate a random maze (when random=True)

        ## Arguments 
        maze (np.array) : a 2D array of 1's and 0's, where 1's represent free cells
        rat (tuple) : a tuple of (row, col) coordinates of the rat
        random_init (bool) : whether to generate a random maze or use the provided maze
        n_rows, n_cols (int) : number of rows and columns in the maze if random=True
        p (float) : the probability of a cell being blocked if random=True
        bombs (int) : number of bombs to randomly place in the maze if random=True
        door_mode (bool) : indicates if bombs play the role of doors or not
        prob_door_close (bool) : probability of a door closing if door_mode=True
        perte (float) : drives the value of the negative reward obtained on risky cells
        """
        if random_init:
            self._maze = self.random_maze(n_rows, n_cols, p, bombs)
        else:
            self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (nrows-1, ncols-1)   # target cell where the "cheese" is
        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.0]
        self.free_cells.remove(self.target)
        self.bomb_cells = [(r,c) for r in range(nrows) for c in range(ncols) if np.abs(self._maze[r,c]+1.)<10**-2]
        self.door_mode = door_mode
        self.prob_door_close = prob_door_close
        self.perte = perte
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not (rat in self.free_cells or rat == self.target or rat in self.bomb_cells):
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.init_pos=rat
        self.reset(rat)
    
    def reset(self, rat):
        self.rat = rat
        self.maze = np.copy(self._maze)
        row, col = rat
        self.maze[row, col] = rat_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        valid_actions = self.valid_actions()
                
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:                  # invalid action, no change in rat position
            nmode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows-1 and rat_col == ncols-1:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if mode == 'invalid':
            return -0.75
        if np.abs(self.maze[rat_row,rat_col]+1.0)<=10**-2 : 
            if self.door_mode:
                return -0.04
            else:
                return -self.perte*np.random.binomial(1,self.prob_door_close)
        if (rat_row, rat_col) in self.visited:
            return -0.25
        
        if mode == 'valid':
            return -0.04

    def new_state_reward(self, action):
        """ 
        Returns the new state and reward after applying the action, without modifiying the maze 
        """
        nrow, ncol, nmode = self.state
        valid_actions = self.valid_actions()
                
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:                  # invalid action, no change in rat position
            nmode = 'invalid'

        nrows, ncols = self.maze.shape
        if nrow == nrows-1 and ncol == ncols-1:
            reward = 1.0
        if nmode == 'blocked':
            reward = self.min_reward - 1
        if nmode == 'invalid':
            reward= -0.75
        if np.abs(self.maze[nrow,ncol]+1.0)<=10**-2 : 
            if self.door_mode:
                reward =  -0.04
            else:
                reward = -self.perte*np.random.binomial(1,self.prob_door_close)
        if (nrow, ncol) in self.visited:
            reward = -0.25
        if nmode == 'valid':
            reward = -0.04
        return nrow, ncol, reward

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = rat_mark
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows-1 and rat_col == ncols-1:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        """
        Returns a list containing the valid actions
        """
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows-1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols-1:
            actions.remove(2)
        r = random.random()
        if row>0 and (self.maze[row-1,col] == 0.0 or 
                    (np.abs(self.maze[row-1,col]+1.)<10**-2 and self.door_mode and r<self.prob_door_close)):
            actions.remove(1)
        if row<nrows-1 and (self.maze[row+1,col] == 0.0 or 
                    (np.abs(self.maze[row+1,col]+1.)<10**-2 and self.door_mode and r<self.prob_door_close)):
            actions.remove(3)
        if col>0 and (self.maze[row,col-1] == 0.0 or 
                    (np.abs(self.maze[row,col-1]+1.)<10**-2 and self.door_mode and r<self.prob_door_close)):
            actions.remove(0)
        if col<ncols-1 and (self.maze[row,col+1] == 0.0 or 
                    (np.abs(self.maze[row,col+1]+1.)<10**-2 and self.door_mode and r<self.prob_door_close)):
            actions.remove(2)
        return actions

    @staticmethod
    def random_maze(nrows, ncols, p, bombs):
        """
        Generate a random maze 
        """
        maze = (np.random.rand(nrows,ncols) < p) * 1.
        maze[0,0] = 1.
        maze[nrows-1,ncols-1] = 1.
        union_find = UnionFind(nrows*ncols)
        for i in range(nrows):
            for j in range(ncols):
                if maze[i,j] == 1.:
                    if i > 0 and maze[i-1,j] == 1.:
                        union_find.union(i*ncols + j,(i-1)*ncols + j)
                    if i < nrows-1 and maze[i+1,j] == 1.:
                        union_find.union(i*ncols + j,(i+1)*ncols + j)
                    if j > 0 and maze[i,j-1] == 1.:
                        union_find.union(i*ncols + j,i*ncols + j-1)
                    if j < ncols-1 and maze[i,j+1] == 1.:
                        union_find.union(i*ncols + j,i*ncols + j+1)
        while not(union_find.connected(0, (nrows-1)*ncols + ncols-1)):
            i,j = np.random.randint(nrows), np.random.randint(ncols)
            if maze[i,j] == 0.:
                maze[i,j] = 1.
                if i > 0 and maze[i-1,j] == 1.:
                        union_find.union(i*ncols + j,(i-1)*ncols + j)
                if i < nrows-1 and maze[i+1,j] == 1.:
                    union_find.union(i*ncols + j,(i+1)*ncols + j)
                if j > 0 and maze[i,j-1] == 1.:
                    union_find.union(i*ncols + j,i*ncols + j-1)
                if j < ncols-1 and maze[i,j+1] == 1.:
                    union_find.union(i*ncols + j,i*ncols + j+1)
        free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if maze[r,c] == 1.0]
        bomb_cells = random.sample(free_cells, bombs)
        for cell in bomb_cells:
            r,c = cell
            maze[r,c] = -1.0
        return maze






                        
class Experience(object):
    """
    ## Description
    Experience storage for deep Q-learning
    """
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = deque([],maxlen=max_memory)
        self.num_actions = num_actions

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Save a transition"""
        Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'game_over'))
        self.memory.append(Transition(*args))

    def predict(self, envstate):
        return self.model.forward(envstate)[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


def show(qmaze):
    """
    Plot qmaze
    """
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    map_dict = {0.:1., 1.:0., -1.:-1.,0.5:0.5}
    map = lambda x: map_dict[x]
    canvas = np.vectorize(map)(canvas)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6
    rat_row, rat_col, _ = qmaze.state
    canvas[rat_row, rat_col] = 0.3   # rat cell
    canvas[nrows-1, ncols-1] = -0.2 # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='RdGy')
    return img

def play_game(model, qmaze, rat_cell, plot=False):
    """
    Play an episode according to the greedy policy derived from model
    """
    qmaze.reset(rat_cell)
    envstate = qmaze.observe()
    while True:
        if(plot):
            show(qmaze)
            plt.show()
        prev_envstate = envstate
        # get next action
        q = model.forward(torch.tensor(prev_envstate, dtype=torch.float))
        action = torch.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            if(plot):
                show(qmaze)
                plt.show()
            return True
        elif game_status == 'lose':
            if(plot):
                show(qmaze)
                plt.show()
            return False

def play_stoch_game(policy_model, qmaze:Qmaze, rat_cell:tuple, plot:bool=False):
    """
    Play an episode according to policy_model
    """
    T=0
    W=0
    qmaze.reset(rat_cell)
    state = qmaze.observe()
    while True:
        if(plot):
            show(qmaze)
            plt.show()
            
        # calculate probabilities of taking each action
        probs = policy_model.forward(torch.tensor(state[0], dtype=torch.float).unsqueeze(0))        
        print(probs)
        # sample an action from that set of probs
        sampler = Categorical(probs)
        action = sampler.sample()

        # use that action in the environment
        new_state, reward, game_status = qmaze.act(action)

        state = new_state
        T+=1
        W+=reward
        if game_status == 'win':
            if(plot):
                show(qmaze)
                plt.show()
            return True
        elif game_status == 'lose':
            if(plot):
                show(qmaze)
                plt.show()
            return False
    
def print_stoch_policy(policy_model, qmaze:Qmaze):
    """
    To vizualise a stochastic policy in the maze environment
    """
    policy = np.copy(qmaze._maze).tolist()
    actions_symbols = {
    LEFT: '\u2190',
    UP: '\u2191',
    RIGHT: '\u2192',
    DOWN: '\u2193',
    }
    nrows, ncols = qmaze._maze.shape
    for row  in range(nrows):
        for col in range(ncols):
            if (row, col) in qmaze.free_cells:
                qmaze.reset((row, col))
                state = qmaze.observe()
                probs = policy_model.forward(torch.tensor(state[0], dtype=torch.float).unsqueeze(0))
                policy[row][col] = actions_symbols[torch.argmax(probs[0]).item()]
            elif np.abs(qmaze._maze[row, col]+1.)<=10**(-2):
                qmaze.reset((row, col))
                state = qmaze.observe()
                probs = policy_model.forward(torch.tensor(state[0], dtype=torch.float).unsqueeze(0))
                policy[row][col] = actions_symbols[torch.argmax(probs[0]).item()]
            elif row == nrows-1 and col == ncols-1:
                policy[row][col] = 'G'
            else:
                policy[row][col] = '#'
    return policy
        
