import json
import numpy as np
from direct_method import fit_q

def load_episodes(path):
    with open(path, 'r') as f:
        return json.load(f)

def doubly_robust(path, gamma=1.0):
    episodes = load_episodes(path)
    model = fit_q(episodes, gamma)
    dr_values = []
    for ep in episodes:
        dr = 0.0
        rho = 1.0
        for t, step in enumerate(ep):
            s, a, r = step['state'], step['action'], step['reward']
            # Q-model prediction
            feat = np.concatenate([np.ravel(s), [a]])
            q_sa = model.predict([feat])[0]
            # Expected Q under π_e
            # assume discrete actions 0..A-1
            q_exp = 0.0
            for a2 in range(model.n_estimators):  # placeholder
                feat2 = np.concatenate([np.ravel(s), [a2]])
                q_exp += step['pi_e'] * model.predict([feat2])[0]
            dr += (gamma**t) * (rho * (r - q_sa) + rho * q_exp)
            rho *= step['pi_e'] / step['pi_b']
        dr_values.append(dr)
    return np.mean(dr_values), np.std(dr_values)/np.sqrt(len(dr_values))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="JSON episodes file")
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()
    mean, stderr = doubly_robust(args.file, args.gamma)
    print(f"DR estimate: {mean:.4f} ± {1.96*stderr:.4f}")
