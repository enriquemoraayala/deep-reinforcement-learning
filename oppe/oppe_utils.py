import os, json, random
import pandas as pd
import torch.nn as nn
import gymnasium as gym
import numpy as np

from ray.rllib.algorithms import Algorithm
from statistics import mean, stdev

def load_checkpoint(checkpoint_path):
        algo = Algorithm.from_checkpoint(checkpoint_path)
        print("Checkpoint loaded")
        policy = algo.get_policy()
        return policy


def load_json_to_df(reader, num_eps):
    rows = []
    # reader = JsonReader(json_path)
    for i in range(num_eps):
        episode = reader.next()
        for step in range(len(episode)):
            row = {'ep': episode['eps_id'][step],
                   'step': step,
                   'obs': episode['obs'][step],
                   'action': episode['actions'][step],
                   'action_prob': episode['action_prob'][step],
                   'logprob': episode['action_logp'][step],
                   'reward': episode['rewards'][step],
                   'next_state': episode['new_obs'][step],
                   'done': episode['dones'][step]
            }
            rows.append(row)
    return pd.DataFrame(rows)

def calculate_value_function(episodes_df, gamma=0.99):
    episodes_idx = episodes_df['ep'].unique()
    eps_return = []
    for ep_idx in episodes_idx:
        episode = episodes_df[episodes_df['ep']==ep_idx]
        ep_return = calculate_return(episode, gamma)
        eps_return.append(ep_return)
    return mean(eps_return), stdev(eps_return)
    

def calculate_return(episode, gamma=0.99):
    """
    Calculates the discounted return for a given episode.

    Args:
        episode (list): A list of dictionaries, where each dictionary represents a step in the episode.
        gamma (float): Discount factor.

    Returns:
        float: The discounted return for the episode.
    """
    G = 0
    t = 0
    for _,step in episode.iterrows():
        G += (gamma**t) * step['reward']
        t = t + 1
    return G


def generate_offline_json(path, agent, num_episodes=50, max_steps=200):
    env = gym.make("LunarLander-v2")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    samples = []
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=random.randint(0, 2**32 - 1))
        done, step = False, 0
        while not done and step < max_steps:
            # BEHAVIOR POLICY: elige al azar (podrías usar otra PPO distinta)
            # action = env.action_space.sample()
            action = agent.compute_single_action(obs, explore=False)
            action, _, info = agent.compute_single_action(
                                    obs,
                                    explore=True,       # usa la distribución estocástica de la policy
                                    full_fetch=True     # devuelve extras, incluido log‑prob
                                )
            action_logp = info["action_logp"]
            action_prob = float(np.exp(action_logp))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            sample = {
                "ep": episode,
                "step": step,
                "obs":        obs.tolist(),
                "action":    int(action),
                "rewards":    float(reward),
                "new_obs":    next_obs.tolist(),
                "dones":      bool(done),
                # probabilidad de la acción bajo la behavior policy (uniforme aquí)
                "action_prob": action_prob,
            }
            samples.append(sample)                
            obs, step = next_obs, step + 1
    with open(path, "w", encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    env.close()

# == Definición del modelo FQE (red neuronal Q) ==
class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_size=64):
        super(QNetwork, self).__init__()
        # Red feed-forward simple: dos capas ocultas y salida de tamaño num_actions
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
    def forward(self, state):
        # Pase hacia adelante. Si el estado es un vector 1D, expandir a 2D (batch de tamaño 1).
        return self.net(state)
    
