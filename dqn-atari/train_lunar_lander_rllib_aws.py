import argparse
import os
from ray.rllib.algorithms.ppo import PPOConfig
import ray
import gymnasium as gym


def main(args):
    # In SageMaker, usually /opt/ml/model is the output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    ray.init()
    env = gym.make("LunarLander-v2")
    config = {'gamma': 0.999,
              'num_workers': 0,
              'monitor': True,
              'framework': 'torch'
              }

    # trainer = ppo.PPO(env=env, config={"env_config": config,})
    trainer = (
                PPOConfig()
                .training(gamma=0.999, lr=0.0001)
                .framework('torch')
                .rollouts(num_rollout_workers=1)
                .resources(num_gpus=0)
                .environment(env="LunarLander-v2")
                .build()
            )
    # algo = config.build()
    for i in range(50):  # 20 iterations
        result = trainer.train()
        print(f"Iteration {i}: reward_mean={result['episode_reward_mean']}")
        # Save checkpoint every 50 iterations
        if i % 10 == 0:
            print(f"saving checkpoint in {output_dir}")
            trainer.save(output_dir)
    # Save final policy
    trainer.save(output_dir)
    ray.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RL agent on Lunar Lander V2 with RLLib")
    parser.add_argument("--local-mode", action="store_true",
                        help="Init Ray in local mode for easier debugging.")
    parser.add_argument("--agent_type", type=str, default='ppo',
                        help="ppo/dqn")
    parser.add_argument("--numgpus", type=str, default='0')
    parser.add_argument('--output_dir', type=str, default='/opt/ml/code/checkpoints/new_checkpoint')
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    print("Ray Version %s" % ray.__version__)
    main(args)
