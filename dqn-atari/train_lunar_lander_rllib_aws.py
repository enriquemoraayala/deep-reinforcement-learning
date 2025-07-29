import argparse
import os
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import ray

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/opt/ml/model')
    return parser.parse_args()

def main():
    args = parse_args()
    # In SageMaker, usually /opt/ml/model is the output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    ray.init()
    config = (
        PPOConfig()
        .environment("LunarLander-v2")
        .framework("torch")
        .rollouts(num_rollout_workers=1)
        .training(train_batch_size=4000)
    )
    algo = config.build()
    for i in range(20):  # 20 iterations
        result = algo.train()
        print(f"Iteration {i}: reward_mean={result['episode_reward_mean']}")
        # Save checkpoint every 5 iterations
        if i % 5 == 0:
            algo.save(output_dir)
    # Save final policy
    algo.save(output_dir)
    ray.shutdown()

if __name__ == "__main__":
    main()
