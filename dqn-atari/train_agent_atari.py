import gym
import random
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
from visual_dqn_agent import Agent


def preprocess_frame(screen, exclude, output):
    """Preprocess Image.
        Params
        ======
            screen (array): RGB Image
            exclude (tuple): Section to be croped (UP, RIGHT, DOWN, LEFT)
            output (int): Size of output image
        """
    # TConver image to gray scale
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

    # Crop screen[Up: Down, Left: right]
    screen = screen[exclude[0]:exclude[2], exclude[3]:exclude[1]]

    # Convert to float, and normalized
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

    # Resize image to 84 * 84
    screen = cv2.resize(screen, (output, output),
                        interpolation=cv2.INTER_AREA)
    return screen


def stack_frame(stacked_frames, frame, is_new):
    """Stacking Frames.

        Params
        ======
            stacked_frames (array): Four Channel Stacked Frame
            frame: Preprocessed Frame to be added
            is_new: Is the state First
        """
    if is_new:
        stacked_frames = np.stack(arrays=[frame, frame, frame, frame])
        stacked_frames = stacked_frames
    else:
        stacked_frames[0] = stacked_frames[1]
        stacked_frames[1] = stacked_frames[2]
        stacked_frames[2] = stacked_frames[3]
        stacked_frames[3] = frame

    return stacked_frames


def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (8, -12, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames


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
    state = stack_frames(None, env.reset(), True)
    eps = eps_start

    action0 = 0  # do nothing
    observation0, reward0, terminal, info = env.step(action0)
    print("Before processing: " + str(np.array(observation0).shape))
    plt.imshow(np.array(observation0))
    plt.show()
    observation0 = preprocess_frame(observation0)
    print("After processing: " + str(np.array(observation0).shape))
    plt.imshow(np.array(np.squeeze(observation0)))
    plt.show()
    action_size = env.action_space.n
    agent = Agent(action_size, seed=0)

    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = stack_frames(None, env.reset(), True)
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            next_state = preprocess_frame(next_state)
            next_state = stack_frames(state, next_state, False)
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
        if np.mean(scores_window) >= 350.0:
            print('\nEnvironment solved in {:d} episodes!\t \
                  Average Score: {:.2f}'.format(i_episode-100,
                                                np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), ckp_path)
            break
    return scores


def train_agent():
    env = gym.make('SpaceInvaders-v0')
    env.reset()
    print(env.action_space)
    print(env.observation_space)
    print(env.env.get_action_meanings())
    scores = cnn_dqn(env, 'checkpoint_visual_atari.pth')
    return scores


if __name__ == '__main__':

    train_agent()
