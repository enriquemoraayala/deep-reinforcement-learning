from unityagents import UnityEnvironment
import gym
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from collections import deque
from visual_dqn_agent import Agent
from setup_logger import logger
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/opt/ml/output/tensorboard/')


def cnn_dqn(env, ckp_path, n_episodes=2000,
            max_t=1000,
            eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy
        action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode)
        for decreasing epsilon
    """
    scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.visual_observations[0]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    print('States look like:')
    plt.imshow(np.squeeze(state))
    plt.show()
    state_size = state.shape
    print('States have shape:', state.shape)

    agent = Agent(state_size, action_size, seed=0)

    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.visual_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)        # save most recent score
        scores.append(score)               # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'
              .format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'
                  .format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), ckp_path)
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\t \
                  Average Score: {:.2f}'.format(i_episode-100,
                                                np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), ckp_path)
            break
    return scores


def train_agent(env_path):
    env = UnityEnvironment(file_name=env_path)
    scores = cnn_dqn(env, 'checkpoint_visual.pth')
    return scores


if __name__ == '__main__':

    parser_sagemaker = argparse.ArgumentParser()
    env = os.environ['SM_TRAINING_ENV']
    parser_sagemaker.add_argument('--hosts', type=list,
                                  default=os.environ['SM_HOSTS'])
    parser_sagemaker.add_argument('--current_host', type=str,
                                  default=os.environ['SM_CURRENT_HOST'])
    parser_sagemaker.add_argument('--model_dir', type=str,
                                  default=os.environ['SM_MODEL_DIR'])
    parser_sagemaker.add_argument('--data_dir', type=str,
                                  default=os.environ['SM_INPUT_DIR'])
    parser_sagemaker.add_argument('--num_gpus', type=int,
                                  default=os.environ['SM_NUM_GPUS'])
    parser_sagemaker.add_argument("--checkpoint_path",
                                  type=str,
                                  default="/opt/ml/checkpoints/",
                                  help="Path where checkpoints will be saved.")

    parser_sagemaker.add_argument('--train', type=str,
                                  default=os.environ['SM_CHANNEL_TRAIN'])

    args_sagemaker = parser_sagemaker.parse_args()

    print(args_sagemaker.data_dir)
    print(os.listdir(args_sagemaker.data_dir))
    print(args_sagemaker.model_dir)
    print(os.listdir(args_sagemaker.model_dir))
    print(args_sagemaker.train)
    print(os.listdir(args_sagemaker.train))
    print(args_sagemaker.checkpoint_path)

    env_path = args_sagemaker.data_dir + '/VisualBanana.app'
    print('env_path %s' % env_path)
    train_agent(env_path)
