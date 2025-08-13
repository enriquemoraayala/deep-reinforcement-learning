# import gym
# import ale_py
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import deque
from visual_dqn_agent import Agent, RandomAgent
import process_frames as pf


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cnn_dqn(env, args, n_episodes=2000,
            max_t=1000,
            eps_start=1.0, eps_end=0.01, eps_decay=0.9999):
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
    ep_steps = []
    scores_window = deque(maxlen=20)  # last 100 scores
    eps = eps_start
    ckp_path = args.model_checkpoint_path
    action0 = 0  # do nothing
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print("Before processing: " + str(np.array(observation).shape))
    plt.imshow(np.array(observation))
    plt.show()
    observation = pf.preprocess_frame(observation, (8, -12, -12, 4), 84)
    print("After processing: " + str(np.array(observation).shape))
    plt.imshow(np.array(np.squeeze(observation)), cmap='gray')
    plt.show()
    action_size = env.action_space.n
    if args.agent_type == 'dqn':
        agent = Agent(action_size, seed=0)
        if args.train_mode == 'resume':
            agent.resume_from_checkpoint(ckp_path, device)
    else:
        agent = RandomAgent(action_size, seed=0)

    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = pf.stack_frames(None, env.reset()[0], True)
        score = 0        
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = pf.stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done or truncated:
                ep_steps.append(t)
                break
        scores_window.append(score)        # save most recent score
        scores.append(score)               # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\nEpisode {}\tAverage Score (last 20 episodes): {:.2f} - \tSteps: {}'
              .format(i_episode, np.mean(scores_window), ep_steps[-1]), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'
                  .format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), './checkpoints/state_dict_0804.pth')
            agent.save_checkpoint(i_episode, score, ckp_path)
        if np.mean(scores_window) >= 350.0:
            print('\nEnvironment solved in {:d} episodes!\t \
                  Average Score: {:.2f}'.format(i_episode,
                                                np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), './checkpoints/state_dict_0804.pth')
            agent.save_checkpoint(i_episode, score, ckp_path)
            break
    return scores, ep_steps


def train_agent(args):
    env = gym.make(args.env, render_mode='rgb_array', full_action_space=False)
    env.reset()
    print(env.action_space)
    print(env.observation_space)
    print(env.env.get_action_meanings())
    print(f'{args.train_mode} a {args.agent_type} Atari Agent')
    scores,steps = cnn_dqn(env, args)
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.plot(np.arange(len(steps)), steps)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RL agent on Atari Games")
    parser.add_argument("--env", type=str,
                        help="Path to configuration file of the envionment.",
                        default='ALE/SpaceInvaders-v5')
    parser.add_argument("--agent_type", help = "dqn/random", default="dqn")
    parser.add_argument("--train_mode", help = "start/resume", default="start" )
    parser.add_argument("--model_checkpoint_path", type=str,
                        help="Path to the model checkpoint",
                        default='./checkpoints/checkpoint_visual_atari_dqn_080424.pth'
                        )
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    train_agent(args)
