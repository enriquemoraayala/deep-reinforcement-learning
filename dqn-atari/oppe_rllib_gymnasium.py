# Gymnasium needs ray 2.10 incompatible with rl-waf. For the Lunar Lander we need to use oppe4rl env
"""Calculating the OPE using RLLIB + V_pi_b + V_pi_e from generated episodes"""

import argparse
import json
import os
from datetime import datetime
import ray
import pandas as pd
import gymnasium as gym
import csv
# import ssl

from ray.tune.registry import register_env
from ray.rllib.algorithms import Algorithm
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.estimators import DoublyRobust, ImportanceSampling, \
                                         DirectMethod, \
                                         WeightedImportanceSampling
from ray.rllib.offline.estimators.fqe_torch_model import FQETorchModel


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


def add_expected_reward_to_df(df, total_episodes):
    discount = 0.99  # based in the PPO original paper, default discount
    for ep in range(total_episodes):
        df_ = df[df['ep'] == ep].copy()
        df_.sort_values(by=['step'], inplace=True)
        cum_reward = 0.0
        j = 0
        for i, step in df_.iterrows():
            cum_reward = cum_reward + step.reward * discount**j
            df.at[i, 'exp_reward'] = cum_reward
            j += 1
    return df


def save_to_csv(args, header, row):
    # Verifica si el archivo ya existe
    file_exists = os.path.isfile(args.real_value_functions_csv)

    # Abre el archivo en modo "append" si existe, o en modo "write" si no
    with open(args.real_value_functions_csv, mode="a" if file_exists else "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
    print(f"Datos escritos en {args.real_value_functions_csv}")


def compute_value_function(agent_type, df):
    J_eps = 0.0
    df_ = df.groupby('ep').last()
    J_eps = df_['exp_reward'].mean()
    print('Total Real Value function of %s Policy: %.8f' %
          (agent_type, J_eps))
    return J_eps

def policy_value_functions(args):
    now = datetime.now()
    now = now.strftime("%d%m%y%H")
    total_episodes_b = int(args.num_beh_episodes)
    df_b = load_json_to_df(args.b_policy_episodes_path, total_episodes_b)
    df_b = add_expected_reward_to_df(df_b, total_episodes_b)
    for exp in range(int(args.e_num_experiments)):
        df_e = load_json_to_df(args.e_policy_episodes_path + f'{str(exp)}', int(args.num_eval_episodes))
        df_e = add_expected_reward_to_df(df_e, int(args.num_eval_episodes))
        v_function_beh = compute_value_function('Behavioral', df_b)
        v_function_eval = compute_value_function('Evaluation', df_e)
        header = ['date', 'num_b_episodes', 'num_e_episodes', 'agent_type_b', 'agent_type_e', 'b_model_version', 'e_model_version',
                'b_real_value_function', 'e_real_value_function']
        row = [now, total_episodes_b,  int(args.num_eval_episodes), args.beh_type, args.agent_type, 'random', args.e_model_version,
            v_function_beh, v_function_eval]
        save_to_csv(args, header, row)
    


def main(args_):
    """Main function"""
    # ray.init(local_mode=args_.local_mode)
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

    policy_value_functions(args_)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RL agent on WAF-Brain Environment with RLLib")
    parser.add_argument("--local-mode", action="store_true",
                        help="Init Ray in local mode for easier debugging.")
    parser.add_argument("--agent_type", default="ppo",
                        help="ppo/dqn/random")
    parser.add_argument("--beh_type", default="random",
                        help="ppo/dqn/random")
    parser.add_argument("--chechpoint_path", default="./checkpoints/130920241043")
    parser.add_argument("--e_model_version", default="130920241043")
    parser.add_argument("--json_path_train", default="./episodes/generated_rllib_random_1000eps_200steps_200424")
    parser.add_argument("--json_path_eval", default="./episodes/generated_rllib_random_300eps_200steps_200424")
    parser.add_argument("--num_beh_episodes", default="5000")
    parser.add_argument("--num_eval_episodes", default="100")
    parser.add_argument("--real_value_functions_csv",
                        default="./results/real_value_functions.csv")
    parser.add_argument("--b_policy_episodes_path",
                        default="./episodes/generated_rllib_random_5000eps_200steps_030524")
    parser.add_argument("--e_policy_episodes_path",
                        default="./episodes/130920241043/181124_generated_rllib_ppo_rllib_seed_0000_100eps_200steps_exp_")
    parser.add_argument("--e_num_experiments",
                        default="20")
    
    
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    main(args)