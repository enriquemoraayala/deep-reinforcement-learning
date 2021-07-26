
# drl_p1_repository

## Udacity Deep Reinforcement Learning Nanodegree P1 Repository

### 1. Introduction

Welcome to my implementation of the Udacity DRL Nanodegree's P1. In this repository you will find:

* Description about the problem to solve
* How to install the environment and dependences
* Implementation details
* How to run and train the experiment
* Final report of the agent performance
* Source code of the implementation in PyTorch and Python 3

### 2. Problem description (the environment)

For this project, we have to train an agent to navigate (and collect bananas!) in a large, square world.

![Image of the environment]
(https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.  
1 - move backward.  
2 - turn left.  
3 - turn right.  

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### 3. How to install the environment and dependences

Follow the instructions below to explore the environment on your own machine! You will also learn how to use the Python API to control your agent.

#### Step 1: Clone the DRLND Repository

First, please follow [the instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

#### Step 2: Download the Unity Environment

For this project, you will **not** need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)  
Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)  
Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)  
Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)  

Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

### 4. Implementation details

I've implemented the basic **DQN** algorithm in order to solve this problem.

You can find a description of this algorithm [here](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287)  

You will find the details of the implementation in the class **Agent** (in the dqn_agent.py file) and the details of the network used in the class **QNetwork** (in the model.py file)

Please, see the **Report.pdf** document for details

### 5. Train and run the Agent

In order to train and/or run the Agent, just follow the **Navigation.ipynb** Jupyter Notebook.  

When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:

```python
env_info = env.reset(train_mode=True)[brain_name]
```





