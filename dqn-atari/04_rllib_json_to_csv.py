import os, json, random, numpy as np, gymnasium as gym
import debugpy
import pandas as pd
import torch
from ray.rllib.offline.json_reader import JsonReader
from oppe_utils import load_json_to_df


def json_to_csv(rllib_json, csv_path):
    beh_reader = JsonReader(rllib_json)
    beh_df = load_json_to_df(beh_reader, 10000)
    beh_df.to_csv(csv_path)


if __name__ == '__main__':

    BEH_EPISODES_JSON_TEST = '/opt/ml/code/episodes/120820251600/011125_01_generated_rllib_ppo_rllib_seed_0000_10000eps_300steps_exp_0'
    CSV_PATH =  '/opt/ml/code/episodes/120820251600/011125_01_generated_rllib_ppo_rllib_seed_0000_10000eps_300steps_exp_0.csv'
    json_to_csv(BEH_EPISODES_JSON_TEST, CSV_PATH)
