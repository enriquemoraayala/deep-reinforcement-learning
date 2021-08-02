import gym
import torch
import process_frames as pf
import matplotlib.pyplot as plt
from matplotlib import animation
from visual_dqn_agent import Agent


def save_frames_as_gif(frames, path='./', filename='gym_animation_1200_episodes.gif'):

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def render_agent(model_path):
    env = init_env()
    action_size = env.action_space.n
    agent = Agent(action_size, seed=0)
    device = torch.device('cpu')
    agent.load_from_checkpoint(model_path, device)
    score = 0
    state = pf.stack_frames(None, env.reset(), True)
    frames = []
    while True:
        frames.append(env.render(mode="rgb_array"))
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        score += reward
        state = pf.stack_frames(state, next_state, False)
        if done:
            print("You Final score is:", score)
            save_frames_as_gif(frames)
            break
        env.close()


def init_env():
    env_name = 'SpaceInvaders-v0'
    env = gym.make(env_name)
    env.reset()
    print('Envirnoment: %s' % env_name)
    print(env.action_space)
    print(env.observation_space)
    print(env.env.get_action_meanings())
    return env


def random_play():
    score = 0
    env = init_env()
    frames = []
    while True:
        frames.append(env.render(mode="rgb_array"))
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            env.close()
            print("Your Score at end of game is: ", score)
            save_frames_as_gif(frames)
            break


if __name__ == '__main__':
    model_path = './checkpoint_visual_atari.pth'
    if model_path == '':
        random_play()
    else:
        render_agent(model_path)
