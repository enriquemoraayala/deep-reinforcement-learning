from os import write
from unityagents import UnityEnvironment
import gym
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from ddpg_agent import Agent
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./tensorboard/')
EPS_START = 1.0
EPS_END = 0.1
LIN_EPS_DECAY = 1e-6


def ddpg(env, ckp_path, n_episodes=1000, max_t=1000,
         print_every=10, save_every=10):

    # agent_number
    agent_number = 0
    # get the default brain
    brain_name = env.brain_names[0]
    print('Brain name: %s' % brain_name)
    brain = env.brains[brain_name]
    print(brain)
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'
          .format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    random_seed = 0

    agent = Agent(state_size, action_size,
                  random_seed, writer)

    scores_deque = deque(maxlen=100)
    scores = []
    eps = EPS_START
    for i_episode in range(1, n_episodes+1):
        # state = env.reset()
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        agent.reset()
        # score = np.zeros(1)
        score = 0
        for t in range(max_t+1):
            action = agent.act(state, eps)
            # next_state, reward, done, _ = env.step(action)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            eps = eps - LIN_EPS_DECAY
            eps = np.maximum(eps, EPS_END)

            if np.any(done):
                break

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'
              .format(i_episode, np.mean(scores_deque)), end="")
        writer.add_scalar('mean last 100 episodes', np.mean(scores_deque))
        writer.add_scalar('reward', score)
        writer.add_scalar('total avg reward', np.mean(scores))

        if i_episode % save_every == 0:
            torch.save(agent.actor_local.state_dict(),
                       'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(),
                       'checkpoint_critic.pth')

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'
                  .format(i_episode, np.mean(scores_deque)))

        if np.mean(scores_deque) > 30.0 and np.mean(scores_deque) <= 31.5:
            print('\nEnvironment solved in {:d} ' +
                  'episodes!\tAverage Score: {:.2f}'
                  .format(i_episode-100, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(),
                       'solved_actor_trained_model.pth')
            torch.save(agent.critic_local.state_dict(),
                       'solved_critic_trained_model.pth')

    return scores


def train_agent(env_path):
    env = UnityEnvironment(file_name=env_path)
    scores = ddpg(env, '')
    return scores


if __name__ == '__main__':

    env_path = '/Users/ESMoraEn/repositories/emoraa-deep-reinforcement-' + \
             'learning/DDPG-Reacher-Unity-Agent/Reacher.app'
    train_agent(env_path)
