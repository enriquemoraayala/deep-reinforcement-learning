import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
import logging
import gym
from ray.rllib.algorithms import Algorithm
from ray.rllib.policy.sample_batch import SampleBatch
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


def is_ppo(args, env, eval_policy, df_b, total_episodes_b, J_eps):
    print('Computing IS OPPE')
    num_experiments = 10
    batch_size = int(total_episodes_b / num_experiments)
    prob_behavior = 1. / env.action_space.n

    errors = []
    iss_oppe = []

    for exp in range(num_experiments):
        print("Starting experiment %d of %d" % (exp, num_experiments))
        episodes = np.random.choice(total_episodes_b, batch_size)
        w_episodes = []

        for ep in episodes:
            df_ = df_b[df_b['ep'] == ep].copy()
            probs = []
            for i, step in df_.iterrows():
                # selecting the state
                obs = torch.from_numpy(step['state'])
                obs = torch.unsqueeze(obs, 0)
                action = step['action']
                logits, _ = eval_policy.model({"obs": obs})
                probs_ = torch.nn.Softmax(dim=1)
                probs_ = probs_(logits)
                prob_ = probs_[0][action]
                prob_behavior = step['prob']
                probs.append(prob_.detach().numpy() / prob_behavior)
                j = int(step['step'])
            w_episodes.append(np.prod(probs) * df_["exp_reward"].iloc[j])

        is_oppe = np.array(w_episodes).mean()
        iss_oppe.append(is_oppe)
        print("Experiment %d of %d: Average IS OPPE for %s %.8f" %
              ((exp+1), num_experiments, args.agent_type, is_oppe))
        error = np.square(is_oppe - J_eps)
        print("Experiment %d of %d: Squared Error for %s %.8f" %
              ((exp+1), num_experiments, args.agent_type, error))
        errors.append(error)

    avg_is = np.array(iss_oppe).mean()
    print("Average IS OPPE for %d episodes: %.8f" % (total_episodes_b,
                                                     avg_is))
    logging.info("Average DM OPPE for %d episodes: %.8f" %
                 (total_episodes_b, avg_is))
    rmse = np.sqrt(np.array(errors).mean())
    std = np.array(errors).std()
    print("RSME IS OPPE for %d Experiments of %d episodes: %.8f" %
          (num_experiments, total_episodes_b, rmse))
    print("STD IS OPPE for %d Experiments of %d episodes: %.8f" %
          (num_experiments, total_episodes_b, std))


def load_algo_from_checkpoint(args):
    if args.agent_type == 'ppo':
        algo_path = f'{args.chechpoint_path}'
        # algo_path = '/home/azureuser/cloudfiles/code/Users/Enrique.Mora/deep-reinforcement-learning/dqn-atari/checkpoints/200420240756/ckpt_ppo_agent_torch_lunar_lander'
        try:
            algo = Algorithm.from_checkpoint(algo_path)
            print('Checkpoint loaded')
            evaluation_policy = algo.get_policy()
        except Exception as e:
            print(e)
            evaluation_policy = None
    return evaluation_policy


def compute_value_function(agent_type, df):
    J_eps = 0.0
    df_ = df.groupby('ep').last()
    J_eps = df_['exp_reward'].mean()
    print('Total Real Value function of %s Policy: %.8f' %
          (agent_type, J_eps))
    return J_eps


def main(args):
    total_episodes_b = int(args.num_beh_episodes)
    env = gym.make("LunarLander-v2")
    eval_policy = load_algo_from_checkpoint(args)
    df_b = load_json_to_df(args.b_policy_episodes_path, int(args.num_beh_episodes))
    df_b = add_expected_reward_to_df(df_b, total_episodes_b)
    df_b.to_csv(args.b_policy_episodes_path_output_csv)
    print(df_b[['reward', 'exp_reward']].head())
    df_e = load_json_to_df(args.e_policy_episodes_path, int(args.num_eval_episodes))
    df_e = add_expected_reward_to_df(df_e, int(args.num_eval_episodes))
    v_function_beh = compute_value_function('Behavioral', df_b)
    v_function_eval = compute_value_function('Evaluation', df_e)
    is_ppo(args, env, eval_policy, df_b, total_episodes_b, v_function_eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute Value function of a trained agend and c")
    parser.add_argument("--env-config", type=str,
                        help="Path to configuration file of the envionment.",
                        default="/Users/ESMoraEn/repositories/"
                                "dcg-oppe/src/config/env.json")
    parser.add_argument("--local-mode", action="store_true",
                        help="Init Ray in local mode for easier debugging.")
    parser.add_argument("--pca_matrix_path", type=str,
                        help="Path to the PCA Matrix for dim reduction",
                        default='/Users/ESMoraEn/repositories/dcg-oppe/'
                                + 'src/torch_pca_reduction_matrix.pt'
                        )
    parser.add_argument("--agent_type", default="ppo")
    parser.add_argument("--dim", default="64")
    parser.add_argument("--chechpoint_path",
                        default="./checkpoints/200420240756/ckpt_ppo_agent_torch_lunar_lander")
    parser.add_argument("--random_episodes_path", default="./outputs_random_2")
    parser.add_argument("--num_beh_episodes", default="5000")
    parser.add_argument("--num_eval_episodes", default="500")
    parser.add_argument("--b_policy_episodes_path",
                        default="./episodes/generated_rllib_random_5000eps_200steps_030524")
    parser.add_argument("--e_policy_episodes_path",
                        default="./episodes/generated_rllib_ppo_rllib_500eps_200steps_030524")
    parser.add_argument("--b_policy_episodes_path_output_csv",
                        default="./episodes/generated_rllib_random_5000eps_200steps_030524.csv")
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    main(args)