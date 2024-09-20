import argparse
import json
import os
import pandas as pd
import ray
import gymnasium as gym

# from ray.rllib.agents.ppo import PPOTrainer
# from ray.rllib.algorithms.dqn.dqn import DQN
from ray import train, tune
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
    config = (
        PPOConfig()
            .environment(env="LunarLander-v2")
            .rollouts(num_rollout_workers=2)  # Number of parallel environments
            .framework("torch")  # Can use "tf" for TensorFlow
            .training(
                gamma=0.99,  # Discount factor
                lr=1e-3,  # Learning rate
                train_batch_size=4000,  # Total training batch size
                sgd_minibatch_size=128,  # Size of minibatches
                num_sgd_iter=10  # Number of SGD iterations per minibatch
            )
            .resources(num_gpus=int(args.numgpus))  # Set this to 1 if you have a GPU   
        )
    config = {'gamma': 0.999,
              'num_workers': 0,
              'monitor': True,
              'framework': 'torch'
              }

    formatted_date = get_file_format()

    columns = ["epoch", "episode_reward_min", "episode_reward_mean",
               "episode_reward_max", "episode_len_mean","num_eps_iter"]
    df_results = pd.DataFrame(columns=columns)
    lengths = []

    checkpoint_dir = '/home/azureuser/cloudfiles/code/Users/Enrique.Mora/deep-reinforcement-learning/dqn-atari/checkpoints/' + formatted_date
    checkpoint_filename_template = '/ckpt_ppo_agent_torch_lunar_lander_gpu_{training_iteration}'
    # Run the training with Tune    
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=train.RunConfig(
            stop={"episode_reward_mean": 200},  # Stop once the average reward reaches 200
            checkpoint_config=train.CheckpointConfig(
                checkpoint_at_end=True  # Save a checkpoint at the end of training
            ),
        ),
    )

    results = tuner.fit()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RL agent on Lunar Lander V2 with RLLib")
    parser.add_argument("--local-mode", action="store_true",
                        help="Init Ray in local mode for easier debugging.")
    parser.add_argument("--agent_type", type=str, default='ppo',
                        help="ppo/dqn")
    parser.add_argument("--numgpus", type=str, default='0')
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    print("Ray Version %s" % ray.__version__)
    main(args)
