# import gym
import gymnasium as gym
import torch
import process_frames as pf
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
from dqn_agent import Agent, RandomAgent
from datetime import datetime

import os
import imageio
from PIL import Image, ImageDraw, ImageFont

from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.algorithms.algorithm import Algorithm

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

def gym2gif(args, env, agent, filename="gym_animation", total_ep=3, max_steps=0):
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
            if args.agent_type == 'random':
                action = env.action_space.sample()
            elif args.agent_type == 'dqn':
                action = agent.getAction(state, epsilon=0)
            elif args.agent_type == 'ppo_rllib':
                action = agent.compute_single_action(state)
            state, reward, done, _, _ = env.step(action)
            score += reward
            if done:
                break
        scores.append(score)
        steps.append(idx_step)
    env.close()
    save_frames_as_gif(frames, path_filename=filename)
    return scores, steps


def generate_episodes(args, env, agent):
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    today = datetime.now()
    today = today.strftime("%d%m%y")
    file_name = f'generated_rllib_{args.agent_type}_{args.total_episodes}eps_{args.max_ep}steps_{today}'
    writer = JsonWriter(
        os.path.join(args.output_episodes, file_name)
    )
    scores = []
    steps = []
    for i in range(int(args.total_episodes)):
        state = env.reset()
        # after reset, state is diferent from env.step() - gymnasium
        state = state[0]
        score = 0
        prev_action = 0
        prev_reward = 0
        if args.max_ep == "0":
            max_steps = 1000
        else:
            max_steps = int(args.max_ep)
        for idx_step in range(max_steps):
            if args.agent_type == 'random':
                action = env.action_space.sample()
            else:
                action = agent.getAction(state, epsilon=0)
            prob, logp = agent.getProbs(state, action)
            next_state, reward, done, _, _ = env.step(action)
            score += reward
            batch_builder.add_values(
                        t=idx_step,
                        eps_id=i,
                        agent_index=0,
                        obs=state,
                        actions=action,
                        action_prob=prob,  # put the true action probability here
                        action_logp=logp,
                        rewards=reward,
                        prev_actions=prev_action,
                        prev_rewards=prev_reward,
                        dones=done,
                        infos=[],
                        new_obs=next_state 
                    )
            prev_action = action
            prev_reward = reward
            state = next_state
            if done:
                break
        scores.append(score)
        steps.append(idx_step)
        writer.write(batch_builder.build_and_reset())
    return scores, steps


def render_agent(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    if args.agent_type == 'dqn':
        agent = Agent(num_states, num_actions)
        agent.load_from_checkpoint(args.model_checkpoint_path, device)
    elif args.agent_type == 'ppo_rllib':
        agent = Algorithm.from_checkpoint(args.model_checkpoint_path)
    elif args.agent_type == 'random':
        agent = RandomAgent(num_actions, 1234)
    if args.render == 'yes':
        scores, steps = gym2gif(args, env, agent, filename=args.output, total_ep=int(args.total_episodes), max_steps=int(args.max_ep))
    else:
        scores, steps = generate_episodes(args, env, agent)
    for i in range(len(scores)):
        print(f'Episode {i} - Score: {scores[i]} - Steps: {steps[i]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                      description="Render RL agent on Atari Games")
    parser.add_argument("--env", type=str,
                        help="Path to configuration file of the envionment.",
                        default='LunarLander-v2')
    parser.add_argument("--agent_type", help = "dqn/random/ppo_rllib", default="random")
    parser.add_argument("--render", help = "yes/no", default="no")
    parser.add_argument("--max_ep", help = "0/max_ep", default="200")
    parser.add_argument("--total_episodes", help = "", default="300")
    parser.add_argument("--output", help = "path", default="./results/gym_lunar_random.gif")
    parser.add_argument("--model_checkpoint_path", type=str,
                        help="Path to the model checkpoint",
                        # default='./checkpoints/checkpoint_lunar_dqn_150424.pth'
                        default='/home/azureuser/cloudfiles/code/Users/Enrique.Mora/deep-reinforcement-learning/dqn-atari/checkpoints/200420240756/ckpt_ppo_agent_torch_lunar_lander'
                        )
    parser.add_argument("--output_episodes", type=str,
                        help="Path to the model checkpoint",
                        default='./episodes'
                        )
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    
    render_agent(args)
