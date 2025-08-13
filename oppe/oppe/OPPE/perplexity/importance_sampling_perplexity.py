import json
import numpy as np

def load_episodes(path):
    with open(path, 'r') as f:
        return json.load(f)

def importance_sampling(path, gamma=1.0):
    episodes = load_episodes(path)
    returns = []
    for ep in episodes:
        rho = 1.0
        G = 0.0
        for t, step in enumerate(ep):
            rho *= step['pi_e'] / step['pi_b']
            G += (gamma ** t) * step['reward']
        returns.append(rho * G)
    return np.mean(returns), np.std(returns) / np.sqrt(len(returns))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="JSON episodes file")
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()
    mean, stderr = importance_sampling(args.file, args.gamma)
    print(f"IS estimate: {mean:.4f} ± {1.96*stderr:.4f}")
