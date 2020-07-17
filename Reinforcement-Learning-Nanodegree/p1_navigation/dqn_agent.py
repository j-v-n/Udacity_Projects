import numpy as np
import random
from collections import deque, namedtuple

from my_model import DQN

import torch
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 64
REPLAY_SIZE = int(1e5)
GAMMA = 0.99
TAU = 0.005
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device("cuda:0")


class DQNAgent:

    """ 
    Class to handle DQN Agent
    
    """

    def __init__(self, state_size, action_size, seed):

        self.seed = random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        self.local_net = DQN(state_size, action_size, seed).to(device)
        self.target_net = DQN(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.local_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(action_size, REPLAY_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) % UPDATE_EVERY == 0:
            if len(self.memory) > 2 * BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            Q_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + gamma * Q_next * (1 - dones)

        Q_local = self.local_net(states).gather(1, actions)

        loss = F.mse_loss(Q_local, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local_net, self.target_net, TAU)

    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def act(self, state, eps=0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.local_net.eval()
        with torch.no_grad():
            action_values = self.local_net(state)
        self.local_net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


class ReplayBuffer:

    """
    Class to handle replaybuffer
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)

        self.batch_size = batch_size

        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                (np.vstack([e.done for e in experiences if e is not None])).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
