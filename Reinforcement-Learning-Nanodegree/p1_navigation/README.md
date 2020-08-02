# Project 1 - Navigation

By Jayanth Nair

### Unity-ML Agents Project
Navigate around the Unity Banana Collector Environment, collecting yellow bananas while avoiding blue bananas using Deep Reinforcement Learning (Windows Environment)

### Project Details
In this project, I train an agent to navigate and collect yellow bananas, while avoiding blue bananas in a Unity environment.
The reward system is as follows:
- +1 for collecting a yellow banana
- -1 for collecting a yellow banana

The state space has 37 dimensions which contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.

The action space has 4 discrete actions which are:
- 0 - move forward
- 1 - move backward
- 2 - turn left
- 3 - turn right

The task is considered solved when the agent scores on average +13 over 100 consecutive episodes. Below is a recording of my agent successfully navigating this environment!

![Solved Model](https://media.giphy.com/media/fSjIbII3F9DS5MIgVu/giphy.gif)
### Getting Started
- The virtual environment used for this task is saved in the environment.yml file.  
- The Unity environment used for this task can be obtained [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip).

### Notebooks and Scripts
- double_dqn_navigation.ipynb -> Jupyter notebook which interacts with Unity Environment and runs training
- my_model.py -> PyTorch implementation of a fully connected neural network 
- my_double_dqn_agent.py -> Defines all functions of the Double DQN Agent
- Report.pdf -> Final report of the project
