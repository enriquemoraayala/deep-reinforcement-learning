# import gym
import gymnasium as gym
import torch
import process_frames as pf
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
from dqn_agent import Agent, RandomAgent

import os
import imageio
from PIL import Image, ImageDraw, ImageFont

def TextOnImg(img, score):
    img = Image.fromarray(img)
    # font = ImageFont.truetype('/Library/Fonts/arial.ttf', 18)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), f"Score={score: .2f}", fill=(255, 255, 255))
    return np.array(img)

def save_frames_as_gif(frames, path_filename):
    print("Saving gif...", end="")
    imageio.mimsave(path_filename, frames, fps=60)

    print("Done!")

def gym2gif(env, agent, filename="gym_animation", total_ep=3, max_steps=0):
    frames = []
    scores = []
    steps = []
    for i in range(total_ep):
        state = env.reset()
        # after reset, state is diferent from env.step()
        state = state[0]
        score = 0
        if max_steps == 0:
            max_steps = 1000
        for idx_step in range(max_steps):
            frame = env.render()
            frames.append(TextOnImg(frame, score))
            action = agent.getAction(state, epsilon=0)
            state, reward, done, _, _ = env.step(action)
            score += reward
            if done:
                break
        scores.append(score)
        steps.append(idx_step)
    env.close()
    save_frames_as_gif(frames, path_filename=filename)
    return scores, steps


def render_agent(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    if args.agent_type == 'dqn':
        agent = Agent(num_states, num_actions)
        agent.load_from_checkpoint(args.model_checkpoint_path, device)
    else:
        agent = RandomAgent(num_actions, 1234)
    if args.render == 'yes':
        scores, steps = gym2gif(env, agent, filename=args.output, total_ep=int(args.total_episodes), max_steps=int(args.max_ep))
    for i in range(len(scores)):
        print(f'Episode {i} - Score: {scores[i]} - Steps: {steps[i]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                      description="Render RL agent on Atari Games")
    parser.add_argument("--env", type=str,
                        help="Path to configuration file of the envionment.",
                        default='ALE/SpaceInvaders-v5')
    parser.add_argument("--agent_type", help = "dqn/random", default="dqn")
    parser.add_argument("--render", help = "yes/no", default="yes")
    parser.add_argument("--max_ep", help = "0/max_ep", default="150")
    parser.add_argument("--total_episodes", help = "", default="10")
    parser.add_argument("--output", help = "path", default="./results/gym_lunar_dqn_150424_150steps.gif")
    parser.add_argument("--model_checkpoint_path", type=str,
                        help="Path to the model checkpoint",
                        default='./checkpoints/checkpoint_lunar_dqn_150424.pth'
                        )
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    
    render_agent(args)
