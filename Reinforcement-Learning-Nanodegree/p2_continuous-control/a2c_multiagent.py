import numpy as np
import random
import copy
from collections import namedtuple, deque
import math

from model import A2CNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


GAMMA = 0.99
# REWARD_STEPS = 2
# BATCH_SIZE = 512
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2



class Agent():



    def __init__(self, state_size, action_size, random_seed,n_agents=20,n_steps=4):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.n_agents = n_agents
        self.n_steps = n_steps
        # Actor Network (w/ Target Network)
        self.net = A2CNetwork(state_size, action_size, random_seed).to(device)
        self.net_optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)



    
    def act(self, state):
        """Returns actions for given state as per current policy."""
        # state = torch.from_numpy(np.expand_dims(state,0)).float().to(device)
        # state = torch.from_numpy(state).float().to(device)
        self.net.eval()
        with torch.no_grad():
            mu_v, var_v, _ = self.net(state)
            mu = mu_v.data.cpu().numpy()
            sigma = torch.sqrt(var_v).data.cpu().numpy()
            action = np.random.normal(mu,sigma)
        self.net.train()
        return np.clip(action, -1, 1)

    def play_n_steps(self,env,brain_name,init_states,episode_end):
        ''' Using a model, collect trajectories over n steps
            Source for code - https://github.com/TomLin/RLND-project/blob/master/p2-continuous-control/a2cModel.py
            This code has been modified to fit my model

            Params:
                model : A2C model 
                env: Unity environment
                brain_name: Unity environment brain
                init_states: Initial states
                n_steps: Number of steps

            Returns:
                states_t : List of states
                actions_t : List of actions
                rewards_t : List of rewards
                dones_t : List of done flags
                init_states : New initial states

        '''
        states_t = []
        actions_t = []
        rewards_t = []
        
        accu_rewards = np.zeros(init_states.shape[0])
        t=0
        states= init_states

        while t<self.n_steps:
            t+=1
            self.net.eval()
            with torch.no_grad():
                states = torch.from_numpy(states).float().to(device)
                actions = self.act(states) 
                

            self.net.train()
            
            
            # env_info = env.step(actions.cpu().data.numpy())[brain_name]
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            rewards = np.array(rewards)
            dones = np.array(dones)
            
            states_t.append(states.cpu().data.numpy())
            actions_t.append(actions)
            rewards_t.append(rewards)
            
            states = next_states
            accu_rewards+= rewards

            if dones.any() or t>=self.n_steps:
                self.net.eval()
                next_states = torch.from_numpy(next_states).float().to(device)
                _, _, values = self.net(next_states)
                values = values.squeeze().detach().cpu().data.numpy()
                
                self.net.train()
                
                for done_id,done in enumerate(dones):
                    if done == True:
                        values[done_id]=0
                
                final_rewards=[]
                rewards_t = np.array(rewards_t)
                
                for reward in rewards_t[::-1]:
                    values = reward+GAMMA*values
                    final_rewards.append(values)
                
                final_rewards = np.array(final_rewards)[::-1]
                break
                
        if dones.any():
            env_info = env.reset(train_mode=True)[brain_name]
            init_states = env_info.vector_observations
            episode_end=True
        
        else:
            init_states = next_states.cpu().data.numpy() 

        states_t = np.stack(states_t)
        actions_t = np.stack(actions_t)


        return states_t, actions_t, final_rewards, init_states, accu_rewards, episode_end


    def learn(self, states_t, actions_t, final_rewards):

        """
        Method for agent to learn

        Code adapted from Maxim Lapan's book

        """

        self.net_optimizer.zero_grad()

        states_t_t = torch.from_numpy(states_t).float().to(device)
        states_t_t = states_t_t.view(-1,states_t.shape[-1])
    
    
        actions_t_t = actions_t.reshape(-1,self.action_size)
        actions_t_t= torch.from_numpy(actions_t_t).float().to(device)
    
        mu_v, var_v, value_v = self.net(states_t_t)
    
    
        final_rewards_t = torch.from_numpy(final_rewards.copy()).float().to(device)
        
        final_rewards_t = final_rewards_t.view(final_rewards_t.shape[0]*final_rewards_t.shape[1])
        
        loss_value_v = F.mse_loss(value_v.squeeze(-1),final_rewards_t)
    
        adv_v =final_rewards_t.unsqueeze(dim=-1)-value_v.detach()
    
        log_prob_v = adv_v*calc_logprob(mu_v,var_v,actions_t_t)
    
        loss_policy_v = -log_prob_v.mean()
    
        ent_v = -(torch.log(2*math.pi*var_v) + 1)/2

        entropy_loss_v = ENTROPY_BETA * ent_v.mean()
    
        loss_v = loss_policy_v + entropy_loss_v + loss_value_v

        loss_v = loss_policy_v+loss_value_v
    
        loss_v.backward()
        self.net_optimizer.step()
       

