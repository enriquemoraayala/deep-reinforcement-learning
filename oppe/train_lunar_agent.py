import torch
import argparse

import numpy as np
import gymnasium as gym

from tqdm import trange
from dqn_agent import Agent
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(env, agent, n_episodes=2000, max_steps=1000, eps_start=1.0,
          eps_end=0.1, eps_decay=0.995, target=200, chkpt=False,
          chkpt_path='./checkpoint/model.pth'):
    score_hist = []
    epsilon = eps_start

    bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'
    # bar_format = '{l_bar}{bar:10}{r_bar}'
    pbar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)
    for idx_epi in pbar:
        state = env.reset()
        state = state[0]
        score = 0
        for idx_step in range(max_steps):
            action = agent.getAction(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.save2memory(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        score_hist.append(score)
        score_avg = np.mean(score_hist[-100:])
        epsilon = max(eps_end, epsilon*eps_decay)

        pbar.set_postfix_str(f"Score: {score: 7.2f}, 100 score avg: {score_avg: 7.2f}")
        pbar.update(0)

        # if (idx_epi+1) % 100 == 0:
        #     print(" ")
        #     sleep(0.1)

        # Early stop
        if len(score_hist) >= 100:
            if score_avg >= target:
                break

    if (idx_epi+1) < n_episodes:
        print("\nTarget Reached!")
    else:
        print("\nDone!")
        
    if chkpt:
        torch.save(agent.net_eval.state_dict(), chkpt_path)

    return score_hist


def plotScore(scores):
    plt.figure()
    plt.plot(scores)
    plt.title("Score History")
    plt.xlabel("Episodes")
    plt.show()


def testLander(env, agent, loop=3):
    for i in range(loop):
        state = env.reset()
        for idx_step in range(500):
            action = agent.getAction(state, epsilon=0)
            env.render()
            state, reward, done, _, _ = env.step(action)
            if done:
                break
    env.close()


def train_agent(args):
    BATCH_SIZE = 128
    LR = 1e-3
    EPISODES = 5000
    TARGET_SCORE = 250.     # early training stop at avg score of last 100 episodes
    GAMMA = 0.99            # discount factor
    MEMORY_SIZE = 10000     # max memory buffer size
    LEARN_STEP = 5          # how often to learn
    TAU = 1e-3              # for soft update of target parameters
    SAVE_CHKPT = True      # save trained network .pth file
    env = gym.make("LunarLander-v2")
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent = Agent(
        n_states = num_states,
        n_actions = num_actions,
        batch_size = BATCH_SIZE,
        lr = LR,
        gamma = GAMMA,
        mem_size = MEMORY_SIZE,
        learn_step = LEARN_STEP,
        tau = TAU,
        )
    score_hist = train(env, agent, n_episodes=EPISODES, target=TARGET_SCORE,
                       chkpt=SAVE_CHKPT, chkpt_path=args.model_checkpoint_path)
    plotScore(score_hist)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RL agent on Atari Games")
    parser.add_argument("--env", type=str,
                        help="Path to configuration file of the envionment.",
                        default="LunarLander-v2")
    parser.add_argument("--agent_type", help = "dqn/random", default="dqn")
    parser.add_argument("--train_mode", help = "start/resume", default="start" )
    parser.add_argument("--model_checkpoint_path", type=str,
                        help="Path to the model checkpoint",
                        default='./checkpoints/checkpoint_lunar_dqn_150424.pth'
                        )
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    train_agent(args)