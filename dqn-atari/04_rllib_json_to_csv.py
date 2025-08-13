import os, json, random, numpy as np, gymnasium as gym
import debugpy
import pandas as pd
import torch
from ray.rllib.offline.json_reader import JsonReader
from oppe_utils import load_json_to_df


def json_to_csv(rllib_json, csv_path):
    beh_reader = JsonReader(rllib_json)
    beh_df = load_json_to_df(beh_reader, 2000)
    beh_df.to_csv(csv_path)


if __name__ == '__main__':

    BEH_EPISODES_JSON_TEST = '/opt/ml/code/episodes/310720251600/310725_generated_rllib_ppo_rllib_seed_0000_2000eps_200steps_exp_7'
    CSV_PATH =  '/opt/ml/code/episodes/310720251600/310725_generated_rllib_ppo_rllib_seed_0000_2000eps_200steps_exp_7.csv'
    json_to_csv(BEH_EPISODES_JSON_TEST, CSV_PATH)
