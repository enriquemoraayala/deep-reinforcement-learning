import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def load_episodes(path):
    with open(path, 'r') as f:
        return json.load(f)

def fit_q(episodes, gamma=1.0):
    X, Y = [], []
    for ep in episodes:
        G = 0.0
        for step in reversed(ep):
            G = step['reward'] + gamma * G
            X.append((step['state'], step['action']))
            Y.append(G)
    # Feature engineering: flatten state and action
    Xf = [np.concatenate([np.ravel(s), [a]]) for s,a in X]
    model = RandomForestRegressor()
    model.fit(Xf, Y)
    return model

def direct_method(path, gamma=1.0):
    episodes = load_episodes(path)
    model = fit_q(episodes, gamma)
    values = []
    for ep in episodes:
        V = 0.0
        for t, step in enumerate(ep):
            feats = np.concatenate([np.ravel(step['state']), []])
            # For discrete actions, enumerate possible actions from data
            # Here assume action space {0,...,A-1}
            qs = []
            for a in range(model.n_estimators):  # placeholder for action set
                feat = np.concatenate([np.ravel(step['state']), [a]])
                qs.append(model.predict([feat])[0] * step.get('pi_e',1.0))
            V += (gamma**t) * sum(qs)
        values.append(V)
    return np.mean(values), np.std(values)/np.sqrt(len(values))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="JSON episodes file")
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()
    mean, stderr = direct_method(args.file, args.gamma)
    print(f"DM estimate: {mean:.4f} ± {1.96*stderr:.4f}")
