import os

import numpy as np
import pandas as pd
import torch
import argparse
import gymnasium as gym

import imageio
from PIL import Image, ImageDraw, ImageFont
from ray.rllib.offline.json_reader import JsonReader


def load_json_to_df(json_path, num_eps):
    rows = []
    reader = JsonReader(json_path)
    for i in range(num_eps):
        episode = reader.next()
        for step in range(len(episode)):
            row = {'ep': episode['eps_id'][step],
                   'step': step,
                   'state': episode['obs'][step],
                   'action': episode['actions'][step],
                   'prob': episode['action_prob'][step],
                   'logprob': episode['action_logp'][step],
                   'reward': episode['rewards'][step],
                   'next_state': episode['new_obs'][step],
                   'done': episode['dones'][step]
            }
            rows.append(row)
    return pd.DataFrame(rows)



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


def gym2gif(env, eps, e, seed, filename="gym_animation", max_steps=0):
    frames = []
    scores = []
    steps = []

    ep = eps[eps['ep'] == e]
    actions = ep['action']
    state = env.reset(seed=seed)
    # after reset, state is diferent from env.step()
    state = state[0]
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
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    eps = load_json_to_df(args.json_file, int(args.num_eps))
    # eps.to_csv(args.json_file + '.csv')
    for e in eps.ep.unique():
        gym2gif(env, eps, e, int(args.env_seed), filename=args.output_file, max_steps=0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize RLLIB Json file with episodes")
    parser.add_argument("--json_file", type=str,
                        help="Path to json file with the episodes.",
                        default="/home/azureuser/cloudfiles/code/Users/Enrique.Mora/ope-dcg/episodes/generated_rllib_random_seed_12345_1eps_200steps_300824")
    parser.add_argument("--output_file",
                        default="/home/azureuser/cloudfiles/code/Users/Enrique.Mora/ope-dcg/episodes/generated_rllib_random_seed_12345_1eps_200steps_300824")
    parser.add_argument("--num_eps", default="1")
    parser.add_argument("--env_seed", default="12345")
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    main(args)