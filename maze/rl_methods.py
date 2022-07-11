import sys
import torch
import numpy as np
import datetime, random
from collections import namedtuple
from tqdm.notebook import trange

sys.path.append('../')
from maze.maze_class import Qmaze, Experience
from src.utils import format_time
from src.network import Net

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'game_over'))

def optimize_model(model:Net, experience:Experience, optimizer:torch.optim, loss_n:torch.nn, batch_size:int, 
                    data_size:int, n_epoch:int):
    """
    ## Description
    Performs one step of Q-learning optimization and returns the loss.
    """
    if len(experience) < data_size:
        return 0.0
    transitions = experience.sample(data_size)
    batch_size = min(data_size, batch_size)
    loss_tot = 0.0
    for _ in range(n_epoch):

        batch = random.sample(transitions, batch_size)
        batch = Transition(*zip(*batch))

        state_batch = torch.cat(batch.state).float()
        next_state_batch = torch.cat(batch.next_state).float()
        reward_batch = torch.stack(batch.reward).float()
        game_over_batch = torch.stack(batch.game_over).float()

        # Compute the state actions values according to the model
        state_action_values = model.forward(state_batch.float())

        # Compute the target state actions values 
        target_state_action_values = model.forward(state_batch.float())
        idx = [np.arange(batch_size), batch.action]
        target_state_action_values[idx] = reward_batch + (game_over_batch == False) * experience.discount * torch.max(model.forward(next_state_batch.float()))

        # Compute loss
        loss = loss_n(state_action_values, target_state_action_values.float()).float()
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tot += loss.detach().item()
    return loss_tot

def qtrain(model:Net, qmaze:Qmaze, optimizer:torch.optim, loss_n:torch.nn, **opt):
    """
    ## Description
    Deep Q-learning with experience replay using the model and the maze.
    ## Parameters
    model (Net) : Neural network to approximate the state action value function
    qmaze (Qmaze) : Considered maze
    optimizer (torch.optim) : Used optimizer
    loss_n (torch.nn) : Loss to optimize
    ## Returns
    Losses, cumulative rewards and win ratios during training
    """

    # Optional parameters
    n_epoch = opt.get('n_epoch', 15000)

    # Parameters for the experience replay
    n_epoch_fit = opt.get('n_epoch_fit', 8)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 32)
    batch_size = opt.get('batch_size', 16)

    name = opt.get('name', '') # to save the parameters
    rat_cell_0 = opt.get("rat_cell_0","") # to reset the maze to (0,0) at every epochs
    
    eps_start = opt.get('eps_start',0.9) 
    eps_end = opt.get('eps_end',0.05)
    eps_decay = opt.get('eps_decay',n_epoch/500)

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory, discount=1.)

    start_time = datetime.datetime.now()
    win_history = [] # to calculate win ratio
    hsize = 10
    win_rate = 0.0
    steps_done = 0

    tab_loss = []
    tab_cumulative_reward = []
    tab_win = []

    for epoch in range(n_epoch):
        loss = 0.0
        rat_cell = random.choice(qmaze.free_cells)
        if rat_cell_0:
            rat_cell = (0,0)
        qmaze.reset(rat_cell)
        game_over = False
        envstate = qmaze.observe()
        n_episodes = 0
        R=0
        while not game_over:
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            # Get next action with epsilon-greedy policy derived from Q with adaptive epsilon
            eps_threshold = eps_end + (eps_start - eps_end) * eps_decay**steps_done
            steps_done += 1
            if np.random.rand() < eps_threshold:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(torch.tensor(prev_envstate, dtype=torch.float)).detach())

            # Apply action, get reward and new envstate
            envstate, reward, game_status = qmaze.act(action)
            R+=reward
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False

            # Store episode (experience)
            experience.push(torch.tensor(prev_envstate, dtype=float), action, torch.tensor(reward, dtype=float), 
                            torch.tensor(envstate, dtype=float), torch.tensor(game_over, dtype=float))
            n_episodes += 1

            # Train neural network model
            loss += optimize_model(model, experience, optimizer, loss_n, batch_size, data_size, n_epoch_fit)

        tab_loss.append(loss/n_episodes)
        tab_cumulative_reward.append(R)

        # Calculate win rate, time and print
        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize
        tab_win.append(win_rate)
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))

    # Save model
    if name:
        model.save(name)

    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, batch_size, t))
    return tab_loss, tab_cumulative_reward, tab_win

def qtrain_change_maze(model:Net, qmaze1:Qmaze, qmaze2:Qmaze, optimizer:torch.optim, loss_n:torch.nn, 
                        change:int, **opt):
    """
    ## Description
    Deep Q-learning using experience replay with a change of environment from qmaze1 to qmaze2 at the end 
    of change epochs
    ## Parameters
    model (Net) : Neural network to approximate the state action value function
    qmaze1 (Qmaze) : First environment
    qmaze2 (Qmaze) : Second environment
    optimizer (torch.optim) : Used optimizer
    loss_n (torch.nn) : Loss to optimize
    change (int) : Number of epochs before the change of environment
    ## Returns
    Losses, cumulative rewards and win ratios during training
    """
    # Optional parameters
    n_epoch = opt.get('n_epoch', 15000)

    # Parameters for the experience replay
    n_epoch_fit = opt.get('n_epoch_fit', 8)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 32)
    batch_size = opt.get('batch_size', 16)

    name = opt.get('name', 'model_parameters') # to save the model
    eps_start = opt.get('eps_start',0.9) 
    eps_end = opt.get('eps_end',0.05)
    eps_decay = opt.get('eps_decay',n_epoch/500)

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory, discount=1.)
    
    qmaze=qmaze1
    start_time = datetime.datetime.now()
    win_history = [] # to calculate win ratio
    hsize = 10
    win_rate = 0.0
    steps_done = 0
    tab_loss = []
    tab_cumulative_reward = []
    tab_win = []
    for epoch in range(n_epoch):
        if epoch == change:
            print("Changing maze")
            qmaze=qmaze2
            steps_done = 0
        loss = 0.0
        rat_cell = random.choice(qmaze.free_cells)
        rat_cell = (0,0)
        qmaze.reset(rat_cell)
        game_over = False
        envstate = qmaze.observe()
        n_episodes = 0
        R=0
        while not game_over:
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            # Get next action with epsilon-greedy policy derived from Q with adaptive epsilon
            eps_threshold = eps_end + (eps_start - eps_end) * eps_decay**steps_done
            steps_done += 1
            if np.random.rand() < eps_threshold:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(torch.tensor(prev_envstate, dtype=torch.float)).detach())

            # Apply action, get reward and new envstate
            envstate, reward, game_status = qmaze.act(action)
            R+=reward
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False

            # Store episode (experience)
            experience.push(torch.tensor(prev_envstate, dtype=float), action, torch.tensor(reward, dtype=float), 
                            torch.tensor(envstate, dtype=float), torch.tensor(game_over, dtype=float))
            n_episodes += 1

            # Train neural network model
            loss += optimize_model(model, experience, optimizer, loss_n, batch_size, data_size, n_epoch_fit)

        tab_loss.append(loss/n_episodes)
        tab_cumulative_reward.append(R)

        # Calculate win rate, time and print
        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize
            tab_win.append(win_rate)
        else:
            tab_win.append(0)
        
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))

    # Save model
    if name:
        model.save(name)

    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, batch_size, t))
    return tab_loss, tab_cumulative_reward, tab_win