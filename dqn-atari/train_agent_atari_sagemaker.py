import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from collections import deque
from visual_dqn_agent import Agent
import process_frames as pf

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
    state = pf.stack_frames(None, env.reset(), True)
    eps = eps_start

    action0 = 0  # do nothing
    observation0, reward0, terminal, info = env.step(action0)
    print("Before processing: " + str(np.array(observation0).shape))
    plt.imshow(np.array(observation0))
    plt.show()
    observation0 = pf.preprocess_frame(observation0, (8, -12, -12, 4), 84)
    print("After processing: " + str(np.array(observation0).shape))
    plt.imshow(np.array(np.squeeze(observation0)), cmap='gray')
    plt.show()
    action_size = env.action_space.n
    agent = Agent(action_size, seed=0)

    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = pf.stack_frames(None, env.reset(), True)
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            next_state = pf.stack_frames(state, next_state, False)
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
        writer.add_scalar('Episode score', score, i_episode)
        writer.add_scalar('Episode steps', t, i_episode)
        writer.add_scalar('Mean score last 100 episodes',
                          np.mean(scores_window), i_episode)
        writer.add_scalar('Max score last 100 episodes', np.max(scores_window),
                          i_episode)

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'
                  .format(i_episode, np.mean(scores_window)))
            logger.info('Episode {}\tAverage Score: {:.2f}'
                        .format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), ckp_path)
        if np.mean(scores_window) >= 450.0:
            print('\nEnvironment solved in {:d} episodes!\t \
                  Average Score: {:.2f}'.format(i_episode-100,
                                                np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), ckp_path)
            break
    return scores


def train_agent(args_sagemaker):
    env_name = 'SpaceInvaders-v0'
    env = gym.make(env_name)
    env.reset()
    logger.info('Starting training. Environment %s loaded' % env_name)
    print(env.action_space)
    print(env.observation_space)
    print(env.env.get_action_meanings())
    scores = cnn_dqn(env, os.path.join(
                                  args_sagemaker.checkpoint_path,
                                  'checkpoint_visual_atari.pth'))
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

    os.system('python -m atari_py.import_roms /opt/ml/input/data/train')
    train_agent(args_sagemaker)
