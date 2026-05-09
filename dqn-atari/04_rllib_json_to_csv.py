import os, json, random, re, numpy as np, gymnasium as gym
import debugpy
import pandas as pd
import torch
from tqdm import tqdm
from ray.rllib.offline.json_reader import JsonReader
from oppe_utils import load_json_to_df_max


def extract_num_eps(path):
    match = re.search(r'_(\d+)eps', os.path.basename(path))
    if match:
        return int(match.group(1))
    raise ValueError(f'No se encontró el número de episodios en el nombre del fichero: {path}')


def json_to_csv(rllib_json, csv_path, num_eps):
    beh_reader = JsonReader(rllib_json)
    beh_df, eps, steps = load_json_to_df_max(beh_reader, num_eps)
    print(f'loaded JSON: {rllib_json}')
    print(f"Transformed {eps} episodes with a total of {steps} steps")
    beh_df.to_csv(csv_path)


if __name__ == '__main__':

    JSON_FILES = [
        #'/opt/ml/code/episodes/310320260800/060426_generated_rllib_ppo_rllib_seed_0000_10000eps_300steps_Truewind_exp_0',
        '/opt/ml/code/episodes/310320260800/060426_generated_rllib_ppo_rllib_seed_0000_2000eps_300steps_Truewind_exp_0',
        #'/opt/ml/code/episodes/310320260800/080426_generated_rllib_ppo_rllib_seed_0000_1000eps_300steps_Truewind_exp_0',
        #'/opt/ml/code/episodes/060420261500/060426_generated_rllib_ppo_rllib_seed_0000_2000eps_300steps_Truewind_exp_0',
    ]

    for json_path in tqdm(JSON_FILES, desc='Convirtiendo ficheros'):
        num_eps = extract_num_eps(json_path)
        csv_path = json_path + '.csv'
        json_to_csv(json_path, csv_path, num_eps)
