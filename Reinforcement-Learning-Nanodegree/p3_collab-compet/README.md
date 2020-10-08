
# Project 3: Collaboration and Competition

### By Jayanth Nair

## Unity-ML Agents Project

Train two competing agents to play a game of tennis against each other using Deep Reinforcement Learning

## Project Details

Reward system:
 - +0.1 if an agent hits the ball over the net
 - -0.01 if an agent lets a ball hit the ground or hits the ball out of bounds

Target score:
 - +0.5 over 100 episodes after taking the maximum over both agents

State space: 
 -  24 variables for each agent corresponding to position and velocity of the ball and racket
 - Each agent receives its own local observation
 
Action size:
 - 2 continuous actions corresponding to movement toward (or away from) the net and jumping

Number of agents:
 - 2 competing agents

Below is a recording of a smart agent trained with the DDPG algorithm!

![SolvedModel](https://media.giphy.com/media/p0XhSa7S3npHmMcaS5/giphy.gif)


### Getting Started

1. The virtual environment used for this task is saved in the environment.yml file 

2. The Unity environment used for this task can be obtained [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    

### Notebooks and Scripts

- model -> Contains neural networks for DDPG
- ddpg_multiagent -> DDPG Agent Class with common actor-critic network
- ddpg_multiagent_v2 -> MADDPG Agent Class
- Tennis Using DDPG.ipynb -> Notebook implementing DDPG
- Tennis Using MADDPG.ipynb -> Notebook implementing MADDPG
- checkpoint_actor.pth -> Solved DDPG weights of actor
- checkpoint_critic.pth -> Solved DDPG weights of critic
- checkpoint_actor_[0/1].pth -> Solved MADDPG actor weights for each agent
- checkpoint_critic_[0/1].pth -> Solved MADDPG critic weights for each agent
- Report.pdf -> Final report