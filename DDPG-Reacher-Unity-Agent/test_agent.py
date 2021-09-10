from os import write
from unityagents import UnityEnvironment
import gym
import random
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from collections import deque
from ddpg_agent import Agent


def load_model(model, path, model_name, device):
    model.load_state_dict(torch.load(os.path.join(path, model_name),
                                     map_location=device))


def test(env, ckp_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    agent = Agent(state_size=33, action_size=4, random_seed=0)

    if ckp_path != '':
        load_model(agent.actor_local, ckp_path,
                   'solved_actor_trained_model.pth', device)
        print('Actor model loaded from %s ' % ckp_path)
        load_model(agent.critic_local, ckp_path,
                   'solved_critic_trained_model.pth', device)
        print('Critic model loaded from %s ' % ckp_path)

    env_info = env.reset(train_mode=False)[brain_name]

    state = env_info.vector_observations
    score = 0.0

    while True:
        action = agent.act(state, 0, False)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[agent_number]
        reward = env_info.rewards[agent_number]
        done = env_info.local_done[agent_number]

        state = next_state
        score += reward

        if done:
            print('\r\tTest Score: {:.2f}'.format(score, end=""))
            break


def test_agent(env_path, ckp_path):
    env = UnityEnvironment(file_name=env_path)
    scores = test(env, '/Users/ESMoraEn/repositories/'
                       'emoraa-deep-reinforcement-learning/'
                       'DDPG-Reacher-Unity-Agent')
    return scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint",
                        help="Relative path to the model checkpoint",
                        type=str, default='./')
    parser.add_argument("-e", "--env_path",
                        help="Complete path to the Reacher Environment",
                        type=str, default='./Reacher.app')

    args = parser.parse_args()

    env_path = args.env_path
    ckp_path = args.checkpoint

    # ckp_path = '/Users/ESMoraEn/repositories/' + \
    #         'emoraa-deep-reinforcement-learning/' + \
    #         'DDPG-Reacher-Unity-Agent'

    # env_path = '/Users/ESMoraEn/repositories/emoraa-deep-reinforcement-' + \
    #            'learning/DDPG-Reacher-Unity-Agent/Reacher.app'

    test_agent(env_path, ckp_path)
