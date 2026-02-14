import os, json, random, numpy as np, gymnasium as gym
import debugpy
import pandas as pd
import torch
from ray.rllib.offline.json_reader import JsonReader
from oppe_utils import load_json_to_df_max


def json_to_csv(rllib_json, csv_path, num_eps):
    beh_reader = JsonReader(rllib_json)
    beh_df, eps, steps = load_json_to_df_max(beh_reader, num_eps)
    print(f'loaded JSON: {rllib_json}')
    print(f"Transformed {eps} episodes with a total of {steps} steps")
    beh_df.to_csv(csv_path)


if __name__ == '__main__':

    BEH_EPISODES_JSON_VAL = '/opt/ml/code/episodes/120820251600/140226_generated_rllib_ppo_rllib_seed_rotate_5eps_300steps_exp_0'
    EVAL_EPISODES_JSON = '/opt/ml/code/episodes/130820251600/140226_generated_rllib_ppo_rllib_seed_rotate_5eps_300steps_exp_0'
    BEH_EPISODES_JSON_TRAIN = '/opt/ml/code/episodes/120820251600/011125_01_generated_rllib_ppo_rllib_seed_0000_10000eps_300steps_exp_0'
    JSON_TO_CONVERT =  EVAL_EPISODES_JSON
    CSV_PATH =  '/opt/ml/code/episodes/130820251600/140226_generated_rllib_ppo_rllib_seed_rotate_5eps_300steps_exp_0.csv'
    NUM_EPS = 5
    json_to_csv(JSON_TO_CONVERT, CSV_PATH, NUM_EPS)
