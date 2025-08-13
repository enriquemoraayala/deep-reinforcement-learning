# import gym
import gymnasium as gym
import torch
import process_frames as pf
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from matplotlib import animation
from visual_dqn_agent import Agent, RandomAgent


def save_frames_as_gif(frames, num, agent_type, path='./', filename='gym_animation_2000_episodes_050424.gif'):

    if agent_type == 'random':
        filename = 'gym_animation_episodes_random_' + str(num) +'.gif'
    else:
        filename = 'gym_animation_episodes_dqn_050424_' + str(num) +'.gif'
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def render_agent(args):
    model_path = args.model_checkpoint_path
    results_df = pd.DataFrame(columns=['ep', 'score', 'steps'])
    env = init_env(args.env)
    action_size = env.action_space.n
    agent = Agent(action_size, seed=0)
    device = torch.device('cpu')
    if args.max_ep == '0':
        max_ep = 10000
    else:
        max_ep = args.max_ep
    # agent.load_from_checkpoint(model_path, device)
    agent.resume_from_checkpoint(model_path, device)
    for i in range(int(args.total_episodes)):
        print('Running agent %d' % i)
        score = 0
        state = pf.stack_frames(None, env.reset()[0], True)
        frames = []
        row = []
        for j in range(max_ep):
            frames.append(env.render())
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            score += reward
            state = pf.stack_frames(state, next_state, False)
            if done:
                print(f'You Final score is: {score} after {j} steps')
                row.append(i)
                row.append(score)
                row.append(j)
                results_df.loc[len(results_df)] = row
                if args.render == 'True':
                    save_frames_as_gif(frames, i, args.agent_type)
                break
    print(f'Avg. Score: {results_df.score.mean()} -  Avg. Steps: {results_df.steps.mean()}')
    env.close()


def init_env(env_name):
    env = gym.make(env_name,render_mode="rgb_array")
    env.reset()
    print('Envirnoment: %s' % env_name)
    print(env.action_space)
    print(env.observation_space)
    print(env.env.get_action_meanings())
    return env


def random_play(args):
    score = 0
    env = init_env(args.env)
    action_size = env.action_space.n
    agent = RandomAgent(action_size, seed=0)
    device = torch.device('cpu')
    results_df = pd.DataFrame(columns=['ep', 'score', 'steps'])
    if args.max_ep == '0':
        max_ep = 10000
    else:
        max_ep = args.max_ep
    frames = []
    for i in range(int(args.total_episodes)):
        print('Running agent %d' % i)
        row = []
        score = 0
        state = pf.stack_frames(None, env.reset()[0], True)
        frames = []
        for j in range(max_ep):
            frames.append(env.render())
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            score += reward
            if done:
                env.close()
                print(f'Your Score is {score} after {j} steps')
                row.append(i)
                row.append(score)
                row.append(j)
                results_df.loc[len(results_df)] = row
                if args.render == 'True':
                    save_frames_as_gif(frames, 0, args.agent_type)
                break
    print(f'Avg. Score: {results_df.score.mean()} -  Avg. Steps: {results_df.steps.mean()}')
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                      description="Render RL agent on Atari Games")
    parser.add_argument("--env", type=str,
                        help="Path to configuration file of the envionment.",
                        default='ALE/SpaceInvaders-v5')
    parser.add_argument("--agent_type", help = "dqn/random", default="random")
    parser.add_argument("--render", help = "yes/no", default="no")
    parser.add_argument("--max_ep", help = "0/max_ep", default="0")
    parser.add_argument("--total_episodes", help = "", default="50")
    parser.add_argument("--output", help = "path", default="./results")
    parser.add_argument("--model_checkpoint_path", type=str,
                        help="Path to the model checkpoint",
                        default='./checkpoints/checkpoint_visual_atari.pth'
                        )
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    
    if args.agent_type == 'random':
        random_play(args)
    else:
        render_agent(args)
