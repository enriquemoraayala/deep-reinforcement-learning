import argparse
import json
import os
import pandas as pd
import ray
import gymnasium as gym

# from ray.rllib.agents.ppo import PPOTrainer
# from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from datetime import datetime


def get_file_format():
    now = datetime.now()
    day = now.strftime("%d")
    month = now.strftime("%m")
    year = now.strftime("%Y")
    hour = now.strftime("%H")
    minute = now.strftime("%M")
    formatted_date = f"{day}{month}{year}{hour}{minute}"
    return formatted_date


def main(args):
    # tf.compat.v1.enable_eager_execution()
    ray.init(local_mode=args.local_mode)
    # run = Run.get_context(allow_offline=True)
    env = gym.make("LunarLander-v2")
    config = {'gamma': 0.999,
              'lr': 0.0001,
              'n_step': 1000,
              'num_workers': 0,
              'monitor': True,
              'framework': 'torch'
              }

    # trainer = ppo.PPO(env=env, config={"env_config": config,})
    trainer = (
                PPOConfig()
                .training(gamma=0.999, lr=0.0001)
                .framework('torch')
                .rollouts(num_rollout_workers=0)
                .resources(num_gpus=0)
                .environment(env="LunarLander-v2")
                .build()
            )

    # rllib_config = PPOConfig().from_dict(config)
    # trainer = rllib_config.build()
    formatted_date = get_file_format()

    columns = ["epoch", "episode_reward_min", "episode_reward_mean",
               "episode_reward_max", "episode_len_mean", "filename"]
    df_results = pd.DataFrame(columns=columns)
    lengths = []
    for n in range(50):
        result = trainer.train()
        if n % 5 == 0:
            print(pretty_print(result))
            if args.agent_type == 'ppo':
                file_name = trainer.save('checkpoints/' +
                                         formatted_date +
                                         '/ckpt_ppo_agent_torch_lunar_lander')
            if args.agent_type == 'dqn':
                file_name = trainer.save('checkpoints/' +
                                         formatted_date +
                                         '/ckpt_dqn_agent_torch_lunar_lander')
        s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"
        lengths += result["hist_stats"]["episode_lengths"]
        print(s.format(
            n,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            file_name
        ))
        new_result = {"epoch": n,
                      "episode_reward_min": result["episode_reward_min"],
                      "episode_reward_mean": result["episode_reward_mean"],
                      "episode_reward_max": result["episode_reward_max"],
                      "episode_len_mean": result["episode_len_mean"],
                      "filename": file_name}
        # df_results = pd.concat([df_results, new_result], ignore_index=True)
        df_results.loc[len(df_results)] = new_result

    print(0)
    print(pretty_print(result))
    df_results.to_csv(f'results/{formatted_date}_{args.agent_type}_lunar_lander.csv',
                      index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RL agent on Lunar Lander V2 with RLLib")
    parser.add_argument("--local-mode", action="store_true",
                        help="Init Ray in local mode for easier debugging.")
    parser.add_argument("--agent_type", type=str, default='ppo',
                        help="ppo/dqn")
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    print("Ray Version %s" % ray.__version__)
    main(args)
