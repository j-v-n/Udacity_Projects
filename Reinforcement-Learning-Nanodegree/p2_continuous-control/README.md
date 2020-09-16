
# Project 2: Continuous Control

By Jayanth Nair

### Unity-ML Agents Project

Train a double-jointed arm to move to target locations using Deep Reinforcement Learning

### Project Details

Reward system:
 - +0.1 for each step that the agent's hand is in the goal location

Target score:
 - +30 over 100 episodes

State space: 
 -  33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 
 
Action size:
 - Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

Number of agents:
 - There are two versions of this project, the first with a single agent, the second with 20 parallel identical agents. I chose the latter for this project.

Below is a recording of a smart agent trained with the DDPG algorithm!

![SolvedModel](https://media.giphy.com/media/lRqJt7KOmwRK8jkkcH/giphy.gif)

### Getting Started

 - The virtual environment used for this task is saved in the environment.yml file

 - The Unity environment used for this task can be obtained [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)


### Notebooks and Scripts

- model -> Contains neural networks for DDPG and A2C
- ddpg_multiagent -> DDPG Agent class
- a2c_multiagent -> A2C Agent class
- Continuous Control_Using DDPG.ipynb -> Training notebook for DDPG Solution
- Continuous Control_Using DDPG.ipynb -> Training notebook for A2C Solution
- DDPG_Play.ipynb -> Notebook to see smart agent trained using DDPG in action
- A2C_Play.ipynb -> Notebook to see smart agent trained using A2C in action
- checkpoint_actor.pth -> Solved actor weights from DDPG solution
- checkpoint_critic.pth -> Solved critic weights from DDPG solution
- a2c_checkpoint.pth -> Solved network weights from A2C solution
- Report.pdf -> Final report

