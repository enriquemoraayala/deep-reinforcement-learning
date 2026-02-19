import os

import numpy as np
import pandas as pd
import torch
import argparse
import gymnasium as gym
from tqdm import trange

import imageio
from PIL import Image, ImageDraw, ImageFont
from ray.rllib.offline.json_reader import JsonReader
from oppe_utils import load_json_to_df_max, reset_env_with_seed


def save_frames_as_gif(frames, path_filename):
    print("Saving gif...", end="")
    imageio.mimsave(path_filename, frames, fps=60)
    print("Done!")


def TextOnImg(img, score):
    img = Image.fromarray(img)
    # font = ImageFont.truetype('/Library/Fonts/arial.ttf', 18)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), f"Score={score: .2f}", fill=(255, 255, 255))
    return np.array(img)


def gym2gif(env, eps, e, state, filename="gym_animation", max_steps=0):
    frames = []
    scores = []
    steps = []
    ep = eps[eps['ep'] == e]
    actions = ep['action']
    score = 0
    for step in ep['step']:
        frame = env.render()
        state, reward, done, _, _ = env.step(actions.iloc[step])
        score += reward
        frames.append(TextOnImg(frame,score))
        if done:
            break
    scores.append(score)
    steps.append(step)
    env.close()
    save_frames_as_gif(frames, path_filename=filename+'_'+str(e)+'.gif')
    return scores, steps




def main(args):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    reader_target = JsonReader(args.json_file)
    eps_df, eps, steps = load_json_to_df_max(reader_target, int(args.num_eps))
    print(f"Transformed {eps} episodes with a total of {steps} steps")
    # eps.to_csv(args.json_file + '.csv')
    i = 0
    for e in eps_df.ep.unique():
        initial_state = reset_env_with_seed(env, i, args.env_seed)
        gym2gif(env, eps_df, e, initial_state, filename=args.output_file, max_steps=0)
        i += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize RLLIB Json file with episodes")
    parser.add_argument("--json_file", type=str,
                        help="Path to json file with the episodes.",
                        default="/opt/ml/code/episodes/130820251600/190226_generated_rllib_ppo_rllib_seed_rotate_15eps_300steps_exp_0")
    parser.add_argument("--output_file",
                        default="/opt/ml/code/output_gifs/130820251600/190226_generated_rllib_ppo_rllib_seed_rotate_15eps_300steps_exp_0")
    parser.add_argument("--num_eps", default="15")
    parser.add_argument("--env_seed", default="rotate")
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    main(args)