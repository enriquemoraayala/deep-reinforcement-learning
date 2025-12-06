"""
This script generates episodes and save them in json rllib format
OR
generates a GIF with the episode
"""

# import gym
import gymnasium as gym
import torch
import process_frames as pf
import matplotlib.pyplot as plt
import argparse
import numpy as np
import math
from datetime import datetime
import ray
import os
import imageio
import tqdm
import debugpy

from tqdm import trange
from PIL import Image, ImageDraw, ImageFont
from gymnasium.wrappers import NormalizeObservation
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.algorithms.algorithm import Algorithm

def TextOnImg(img, score, x=20, y=20, text='Score'):
    img = Image.fromarray(img)
    # font = ImageFont.truetype('/Library/Fonts/arial.ttf', 18)
    draw = ImageDraw.Draw(img)
    draw.text((x, y), f"{text}={score: .2f}", fill=(255, 255, 255))
    return np.array(img)

def save_frames_as_gif(frames, path_filename):
    print("Saving gif...", end="")
    imageio.mimsave(path_filename, frames, fps=60)

    print("Done!")

def gym2gif(args, env, agent, filename="gym_animation", total_ep=3, max_steps=0):
    scores = []
    steps = []
    for i in range(total_ep):
        frames = []
        state = env.reset()
        # after reset, state is diferent from env.step()
        state = state[0]
        score = 0
        frame = env.render()
        frames.append(TextOnImg(frame, score, text='Score'))
        if max_steps == 0:
            max_steps = 1000
        for idx_step in range(max_steps): 
            if args.agent_type == 'random':
                action = env.action_space.sample()
            elif args.agent_type == 'dqn':
                action = agent.getAction(state, epsilon=0)
            elif args.agent_type == 'ppo_rllib':
                action = agent.compute_single_action(state, explore=False)
            state, reward, done, _, _ = env.step(action)
            score += reward
            frame = env.render()
            frames.append(TextOnImg(frame, score, text='Score'))
            # frames.append(TextOnImg(frame, idx_step, 20, 80, text='Step' ))
            if i > 200:
                pass
            if done:
                break
        scores.append(score)
        steps.append(idx_step)
        filename_ = filename + f'_{i}.gif'
        print(f"saving {filename_}")
        save_frames_as_gif(frames, path_filename=filename_)
    env.close()
    
    return scores, steps


def generate_episodes(args, env, agent, exp):
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    today = datetime.now()
    today = today.strftime("%d%m%y")

    if args.agent_type == 'random':
        file_name = f'{today}_generated_rllib_{args.agent_type}_seed_{args.env_seed}_{args.total_episodes}eps_{args.max_ep}steps_exp_{exp}'
    else:
        file_name = f'{today}_generated_rllib_{args.agent_type}_seed_{args.env_seed}_{args.total_episodes}eps_{args.max_ep}steps_exp_{exp}'
    writer = JsonWriter(
        os.path.join(args.output_episodes, file_name)
    )
    scores = []
    steps = []
    for i in trange(int(args.total_episodes)):
        if args.env_seed == '0000':
            state = env.reset()
        else:
            state = env.reset(seed=int(args.env_seed))
        # after reset, state is diferent from env.step() - gymnasium
        state = state[0]
        score = 0
        prev_action = 0
        prev_reward = 0
        if args.max_ep == "0":
            max_steps = 1000
        else:
            max_steps = int(args.max_ep)
        for idx_step in range(max_steps):
            if args.agent_type == 'random':
                action = env.action_space.sample()
                prob, logp = agent.getProbs(state, action)
            elif args.agent_type == 'dqn':
                action = agent.getAction(state, epsilon=0)
                prob, logp = agent.getProbs(state, action)
            elif args.agent_type == 'ppo_rllib':
                action = agent.compute_single_action(state, explore=False)
                # action_2, state_2, extra_info = agent.compute_single_action(state, explore=False, full_fetch=True )
                state = torch.from_numpy(np.stack(state))
                state = torch.unsqueeze(state, 0)
                policy = agent.get_policy()
                logits, _ = policy.model({"obs": state})
                probs_ = torch.nn.Softmax(dim=1)
                probs_ = probs_(logits)
                prob = probs_[0][action]
                prob = prob.detach().numpy()
                logp = math.log(prob)
            
            next_state, reward, terminated, truncated , info = env.step(action)
            score += reward
            batch_builder.add_values(
                        t=idx_step,
                        eps_id=i,
                        agent_index=0,
                        obs=state,
                        actions=action,
                        action_prob=prob,  # put the true action probability here
                        action_logp=logp,
                        rewards=reward,
                        prev_actions=prev_action,
                        prev_rewards=prev_reward,
                        terminateds=terminated,
                        truncateds=truncated,
                        infos=info,
                        new_obs=next_state 
                    )
            prev_action = action
            prev_reward = reward
            state = next_state
            if terminated or truncated:
                break
        scores.append(score)
        steps.append(idx_step)
        writer.write(batch_builder.build_and_reset())
    return scores, steps


def render_agent(args):
    num_experiments = int(args.total_datasets_to_generate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_eps = int(args.max_ep)
    if max_eps > 0:
        env = gym.make("LunarLander-v3", render_mode="rgb_array", max_episode_steps=int(args.max_ep))
    else:
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
    # env = NormalizeObservation(env)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    if args.agent_type == 'dqn':
        pass
        # agent = Agent(num_states, num_actions)
        # agent.load_from_checkpoint(args.model_checkpoint_path, device)
    elif args.agent_type == 'ppo_rllib':
        # agent = Algorithm.restore(checkpoint_path=args.model_checkpoint_path)
        agent = Algorithm.from_checkpoint(args.model_checkpoint_path)
        print("Checkpoint loaded")
    elif args.agent_type == 'random':
        # agent = RandomAgent(num_actions, 1234)
        pass
    print(f"Generating {num_experiments} experiments")
    for experiment in range(num_experiments):
        if args.render == 'yes':
            scores, steps = gym2gif(args, env, agent, filename=args.output, total_ep=int(args.total_episodes), max_steps=int(args.max_ep))
        else:
            scores, steps = generate_episodes(args, env, agent, experiment)
            print(f'Experiment {experiment} - Num Episodes {args.total_episodes} - ' +
                  f'Avg. Score: {np.mean(scores)} - Avg. Steps: {np.mean(steps)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                      description="Render RL agent on Atari Games")
    parser.add_argument("--env", type=str,
                        help="Path to configuration file of the envionment.",
                        default='LunarLander-v3')
    parser.add_argument("--agent_type", help = "dqn/random/ppo_rllib", default="ppo_rllib")
    parser.add_argument("--render", help = "yes/no", default="no")
    parser.add_argument("--max_ep", help = "0 is max_ep", default="300")
    parser.add_argument("--total_episodes", help = "", default="5000")
    parser.add_argument("--total_datasets_to_generate", help = "", default="1")
    parser.add_argument("--env_seed", help = "0000 -> no seed", default="0000")
    parser.add_argument("--debug", help = "yes=1/no=0", default="1")
    parser.add_argument("--output", help = "path", 
                        # default="/home/enrique/repositories/deep-reinforcement-learning/dqn-atari/episodes/ppo_rllib_130920241043"
                        default="/opt/ml/code/output_gifs/130820251600_"
                        )
    parser.add_argument("--model_checkpoint_path", type=str,
                        help="Path to the model checkpoint",
                        # default='/home/azureuser/cloudfiles/code/Users/Enrique.Mora/deep-reinforcement-learning/dqn-atari/checkpoints/checkpoint_lunar_dqn_150424.pth'
                        # default='/home/enrique/repositories/deep-reinforcement-learning/dqn-atari/checkpoints/130920241043/ckpt_ppo_agent_torch_lunar_lander'
                        default='/opt/ml/code/checkpoints/120820251600'
                        )
    parser.add_argument("--output_episodes", type=str,
                        #default='/home/enrique/repositories/deep-reinforcement-learning/dqn-atari/episodes/130920241043'
                        default='/opt/ml/code/episodes/120820251600'
                        )
    args = parser.parse_args()

    if args.debug == '1':
        # Escucha en el puerto 5678 (puedes cambiarlo)
        debugpy.listen(("0.0.0.0", 5678))
        print("Esperando debugger de VS Code para conectar...")
        debugpy.wait_for_client()

    print(f"Running with following CLI options: {args}")
    print(ray.__version__)
    
    render_agent(args)
