# Gymnasium needs ray 2.10 incompatible with rl-waf. For the Lunar Lander we need to use oppe4rl env
"""Calculating the OPE using RLLIB"""

import argparse
import json
import os
from datetime import datetime
import ray
import pandas as pd
import gymnasium as gym
# import ssl

from ray.tune.registry import register_env
from ray.rllib.algorithms import Algorithm
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.estimators import DoublyRobust, ImportanceSampling, \
                                         DirectMethod, \
                                         WeightedImportanceSampling
from ray.rllib.offline.estimators.fqe_torch_model import FQETorchModel


def main(args_):
    """Main function"""
    ray.init(local_mode=args_.local_mode)
    # env = gym.make("LunarLander-v2")

    if args_.agent_type == 'ppo':
        path_to_checkpoint = f'{args_.chechpoint_path}/' +\
                             'ckpt_ppo_agent_torch_lunar_lander'
        algo = Algorithm.from_checkpoint(path_to_checkpoint)
    if args_.agent_type == 'dqn':
        path_to_checkpoint = f'{args_.chechpoint_path}/' +\
                             'ckpt_dqn_agent_torch_pca/checkpoint_000006'
        algo = Algorithm.from_checkpoint(path_to_checkpoint)
    if args_.agent_type == 'random':
        algo = 'random'

    dr_estimator = DoublyRobust(
        policy=algo.get_policy(),
        gamma=0.99,
        q_model_config={"type": FQETorchModel, "n_iters": 10},
        )

    is_estimator = ImportanceSampling(
        policy=algo.get_policy(),
        gamma=0.99,
        epsilon_greedy=0.05
    )

    wis_estimator = WeightedImportanceSampling(
        policy=algo.get_policy(),
        gamma=0.99,
        epsilon_greedy=0.05
    )

    dm_estimator = DirectMethod(
        policy=algo.get_policy(),
        gamma=0.99,
        q_model_config={"type": FQETorchModel, "n_iters": 100},
    )

    # Train estimator's Q-model; only required for DM and DR estimators
    reader_train = JsonReader(args_.json_path_train)
    print('Training DM estimator')
    for _ in range(1000):
       batch = reader_train.next()
       print(dm_estimator.train(batch))

    reader_train = JsonReader(args_.json_path_train)
    print('Training DR estimator')
    for _ in range(1000):
       batch = reader_train.next()
       print(dr_estimator.train(batch))
       

    columns = ['v_behavior', 'v_behavior_std', 'v_target', 'v_target_std',
               'v_gain', 'v_delta']
    df_results_dr = pd.DataFrame(columns=columns)
    df_results_is = pd.DataFrame(columns=columns)
    df_results_wis = pd.DataFrame(columns=columns)
    df_results_dm = pd.DataFrame(columns=columns)
    # reader = JsonReader(args.json_path)
    # Compute off-policy estimates
    reader_eval = JsonReader(args_.json_path_eval)
    i = 0
    for _ in range(300):
        batch = reader_eval.next()
        print(dr_estimator.estimate(batch))
        row = pd.DataFrame([dr_estimator.estimate(batch)])
        df_results_dr = pd.concat([df_results_dr, row], ignore_index=True)
        row = pd.DataFrame([is_estimator.estimate(batch)])
        df_results_is = pd.concat([df_results_is, row], ignore_index=True)
        row = pd.DataFrame([wis_estimator.estimate(batch)])
        df_results_wis = pd.concat([df_results_wis, row], ignore_index=True)
        row = pd.DataFrame([dm_estimator.estimate(batch)])
        df_results_dm = pd.concat([df_results_dm, row], ignore_index=True)
        if i % 20 == 0:
            print(df_results_dr[i-10:i])
        i += 1

    now = datetime.now()
    now = now.strftime("%d%m%y%H")
    df_results_dr.to_csv('./results/' +
                         f'{now}_e_{args_.agent_type}_b_{args_.beh_type}_dr.csv',
                         )
    df_results_is.to_csv('./results/' +
                         f'{now}_e_{args_.agent_type}_b_{args_.beh_type}_is.csv',
                         )
    df_results_wis.to_csv('./results/' +
                          f'{now}_e_{args_.agent_type}_b_{args_.beh_type}_wis.csv',
                          )
    df_results_dm.to_csv('./results/' +
                         f'{now}_e_{args_.agent_type}_b_{args_.beh_type}_dm.csv',
                         )

    print('DM V_behavior: %0.3f' % df_results_dm.v_behavior.mean())
    print('DM V_target: %0.3f' % df_results_dm.v_target.mean())
    print('IS V_behavior: %0.3f' % df_results_is.v_behavior.mean())
    print('IS V_target: %0.3f' % df_results_is.v_target.mean())
    print('WIS V_behavior: %0.3f' % df_results_wis.v_behavior.mean())
    print('WIS V_target: %0.3f' % df_results_wis.v_target.mean())
    print('DR V_behavior: %0.3f' % df_results_dr.v_behavior.mean())
    print('DR V_target: %0.3f' % df_results_dr.v_target.mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RL agent on WAF-Brain Environment with RLLib")
    parser.add_argument("--local-mode", action="store_true",
                        help="Init Ray in local mode for easier debugging.")
    parser.add_argument("--agent_type", default="ppo",
                        help="ppo/dqn/random")
    parser.add_argument("--beh_type", default="random",
                        help="ppo/dqn/random")
    parser.add_argument("--chechpoint_path", default="./checkpoints/200420240756")
    parser.add_argument("--json_path_train", default="/home/azureuser/cloudfiles/code/Users/Enrique.Mora/deep-reinforcement-learning/dqn-atari/episodes/generated_rllib_random_1000eps_200steps_200424")
    parser.add_argument("--json_path_eval", default="/home/azureuser/cloudfiles/code/Users/Enrique.Mora/deep-reinforcement-learning/dqn-atari/episodes/generated_rllib_random_300eps_200steps_200424")
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    main(args)