# DDPG Agent

DDPG agent for the Unity Reacher environment, one single agent version.

!['One single arm problem. DDPG agent solved'](https://media.giphy.com/media/W1BmWrC2iYe6GVMvpx/giphy.gif?cid=790b76119d0c293b5fa6672afdf613e01fb943f1fe79831f&rid=giphy.gif&ct=g)

## Project Details

The environment of this project has 4 possible actions corresponding to torque applicable to two joints (each with value in [-1,1]).
The state space has 33 dimensions and contains position, rotation, velocity, and angular velocities of the arm
The environment reward system is +0.1 each time the arm touches the target.
The environment is considered solved when the agent achieves an average reward of +30 (over 100 consecutive episodes).

## Getting Started

Make sure you have python installed in your computer. The project was created with python 3.6.7 (anaconda environmnet)

Clone the repository

`git clone ....`

Navigate to the root of the project:

`cd DDPG-Reacher-Unity-Agent`

Install required python packages using pip or conda, for a quick basic setup use:

`pip install -r requirements.txt`

The repo does not contain any version of the unity environment (Udacity version) otherwise you would need to download it and place it under the Unity folder.

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

## Instructions

You can run the project from some Editor like VS code or directly from commandline:

`python train_agent.py`

This will train the agent and will store 2 versions of model weights. One when it pass the environment solved condition and other after the training episodes.

it stores the model in solved_actor_trained_model.pth and solved_critic_trained_model.pth on the root of the project.

Once the model is trained you can check its behavior by testing it:

`python test_agent.py -c <<checkpoint_directory_path>> -e <<complete_environment_path>>`
