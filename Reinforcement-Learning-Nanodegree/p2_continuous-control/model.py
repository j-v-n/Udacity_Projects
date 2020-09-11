import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()   
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))



class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
       
        self.fcs1 = nn.Linear(state_size, fcs1_units)
    
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
  
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)



class D4PGCritic(nn.Module):

    """ Critic Model for D4PG Algorithm"""


    def __init__(self,state_size,action_size,seed,n_atoms,v_min,v_max,fcs1_units=128,fc2_units=64):
        super(D4PGCritic,self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fcs1 = nn.Linear(state_size,fcs1_units)

        self.fc2 = nn.Linear(fcs1_units+action_size,fc2_units)

        self.fc3 = nn.Linear(fc2_units,n_atoms)
        
        delta = (v_max-v_min)/(n_atoms-1)

        self.register_buffer('supports',torch.arange(v_min,v_max+delta,delta)) # from Maxim Lapan's book Deep RL Hands on. Used to get from the probability distribution to the single mean Q-value
        
        self.reset_parameters()
    

    def forward(self,state,action):
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs,action),dim=1)
        x = F.relu(self.fc2(x))

        return self.fc3(x)

    
    def distr_to_q(self,distr):
        weights = F.softmax(distr,dim=1)*self.supports #
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
