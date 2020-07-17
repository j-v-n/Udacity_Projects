import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    """
    Class for deep Q networks

    Inputs
        - state size
        - action size


    Outputs
        - action value predictions for each action given a particular state
    """

    def __init__(self, state_size, action_size, seed):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state):
        return self.fc(state)
