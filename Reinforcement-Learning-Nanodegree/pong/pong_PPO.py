import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt 

import pong_utils
import atari_py
import gym
from parallelEnv import parallelEnv



import time



RIGHT=4
LEFT=5
class Policy(nn.Module):

    def __init__(self, input_shape):
        super(Policy, self).__init__()
        
    
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        # 1 fully connected layer
        self.fc = nn.Sequential(nn.Linear(conv_out_size, 64), nn.ReLU(),nn.Linear(64,1))
        self.sig = nn.Sigmoid()
    
    def _get_conv_out(self,shape):
        o = self.conv(torch.zeros(1,*shape))
        return int(np.prod(o.size()))
        
    def forward(self, x):

        x_size = x.size()
        conv_out = self.conv(x).view(x.size()[0],-1)

        return self.sig(self.fc(conv_out))



def surrogate(policy, old_probs, states, actions, rewards,
              discount = 0.995, beta=0.01,epsilon=0.1):

    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = pong_utils.states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)

    reweighting_factor = new_probs/old_probs
    
    clipped = torch.clamp(reweighting_factor,1-epsilon,1+epsilon)

    clipped_surrogate = torch.min(reweighting_factor*rewards,clipped*rewards)
    
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))


    return torch.mean(clipped_surrogate + beta*entropy)



if __name__=='__main__':

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = gym.make('PongDeterministic-v4')

    policy = Policy((2,80,80)).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=1e-4)


    envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)


    discount_rate = 0.99
    beta = 0.01
    tmax = 320
    epsilon = 0.1
    SGD_EPOCH = 5


    mean_rewards=[]

    episode = 5000
    for e in range(episode):

        # collect trajectories
        old_probs, states, actions, rewards = \
            pong_utils.collect_trajectories(envs, policy, tmax=tmax)
            
        total_rewards = np.sum(rewards, axis=0)

        for _ in range(SGD_EPOCH):
            L = -surrogate(policy,old_probs,states,actions,rewards,epsilon=epsilon,beta=beta)   
  
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
        del L
            
        
        epsilon *= .999
        
        # the regulation term also reduces
        # this reduces exploration in later runs
        beta*=.995
        
        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))
        
        # display some progress every 20 iterations
        if (e+1)%20 ==0 :
            print("Episode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))
            print(total_rewards)

    plt.plot(mean_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Rewards')
    plt.savefig('PPO_results.png')

    torch.save(policy,'ppo_solved.policy')