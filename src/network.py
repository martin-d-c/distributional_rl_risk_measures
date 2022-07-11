import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)

class Net(nn.Module):
    def __init__(self,size, num_actions):
      super(Net, self).__init__()
      self.network = nn.Sequential(
          nn.Linear(size, size),
          nn.PReLU(),
          nn.Linear(size, size),
          nn.PReLU(),
          nn.Linear(size, num_actions)
        ).float()
      self.apply(initialize_weights)

    def forward(self, x):
      return self.network(x)
    
    def load(self, path):
      self.network.load_state_dict(torch.load(path))
    
    def save(self, name):
      torch.save(self.network.state_dict(),'../parameters/model_parameters/' +name+ '.pt')


class PolicyNet(nn.Module):
  def __init__(self, size, num_actions):
    super(PolicyNet, self).__init__()
    self.network = nn.Sequential(
        nn.Linear(size, size//2),
          nn.PReLU(),
          nn.Linear(size//2, size//4),
          nn.PReLU(),
          nn.Linear(size//4, num_actions)
      ).float()
    self.apply(initialize_weights)
    self.num_actions = num_actions
    self.size = size
  
  def forward(self, state):
    return F.softmax(self.network(state), dim=1)

  def load(self, path):
    self.network.load_state_dict(torch.load(path))
  
  def save(self, name):
    torch.save(self.network.state_dict(), '../parameters/policy_parameters/'+name+ '.pt')  
  
class PolicyNetPortfolio(nn.Module):
  def __init__(self, size, num_actions):
    super(PolicyNetPortfolio, self).__init__()
    self.network = nn.Sequential(
        nn.Linear(size, size*4),
        nn.Linear(size*4, num_actions)
      ).float()
    self.apply(initialize_weights)
    self.num_actions = num_actions
    self.size = size
  
  def forward(self, state):
    return F.softmax(self.network(state), dim=0)

  def load(self, path):
    self.network.load_state_dict(torch.load(path))
  
  def save(self, name):
    torch.save(self.network.state_dict(), '../parameters/policy_parameters/'+name+ '.pt')  


  