"""
Off‑Policy Policy Evaluation con RLlib 2.11
==========================================

• Se entrena (rápidamente) una PPO ‑> será la **política objetivo** (π_target).
• Se crea una *behavior policy* distinta y se generan episodios offline (.json).
• Se entrena un Q‑model para los estimadores que lo necesitan (DM y DR).
• Se instancian y entrenan DM, IS y DR con los datos offline.
• Se estiman los retornos de π_target aprovechando los estimadores entrenados.
"""

import os, json, random, numpy as np, gymnasium as gym
import debugpy

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.offline.json_reader import JsonReader

from ray.rllib.algorithms import Algorithm
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.estimators import DoublyRobust, ImportanceSampling, \
                                         DirectMethod, \
                                         WeightedImportanceSampling
from ray.rllib.offline.estimators.fqe_torch_model import FQETorchModel


#from ray.rllib.offline.estimators.direct_method import DMEstimator
#from ray.rllib.offline.estimators.importance_sampling import ImportanceSamplingEstimator as ISEstimator
#from ray.rllib.offline.estimators.doubly_robust import DoublyRobustEstimator as DREstimator
import ray
ray.init(ignore_reinit_error=True, include_dashboard=False)
print(ray.__version__)

debug = 0

if debug == 1:
    # Escucha en el puerto 5678 (puedes cambiarlo)
    debugpy.listen(("0.0.0.0", 5678))
    print("Esperando debugger de VS Code para conectar...")
    debugpy.wait_for_client()


def load_checkpoint(checkpoint_path):
        algo = Algorithm.from_checkpoint(checkpoint_path)
        print("Checkpoint loaded")
        policy = algo.get_policy()
        return policy

# --------------------------------------------------------------------------- #
# 2. GENERAR DATASET OFFLINE CON UNA BEHAVIOR POLICY DISTINTA
# --------------------------------------------------------------------------- #
def generate_offline_json(path, agent, num_episodes=50, max_steps=200):
    env = gym.make("LunarLander-v2")
    os.makedirs(os.path.dirname(path), exist_ok=True)
   
    with open(path, "w") as f:
        for _ in range(num_episodes):
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
                    "obs":        obs.tolist(),
                    "actions":    int(action),
                    "rewards":    float(reward),
                    "new_obs":    next_obs.tolist(),
                    "dones":      bool(done),
                    # probabilidad de la acción bajo la behavior policy (uniforme aquí)
                    "action_prob": action_prob,
                }
                f.write(json.dumps(sample) + "\n")
                obs, step = next_obs, step + 1
    env.close()

BEH_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/310720251600"
EVAL_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/300720251000"
DATA_DIR = "/opt/ml/code/episodes/selfcontained"
TRAIN_JSON = f"{DATA_DIR}/train.json"
TEST_JSON  = f"{DATA_DIR}/test.json"
beh_agent = load_checkpoint(BEH_CHECKPOINT_PATH)
eval_agent = load_checkpoint(EVAL_CHECKPOINT_PATH)
generate_offline_json(TRAIN_JSON, beh_agent, num_episodes=100)
generate_offline_json(TEST_JSON, beh_agent, num_episodes=30)

# --------------------------------------------------------------------------- #
# 3. ENTRENAR ESTIMADOR DM (requiere Q‑model)
# --------------------------------------------------------------------------- #
reader_train = JsonReader(TRAIN_JSON)  # 200 transiciones/batch
dm_est = DirectMethod(policy=eval_agent,
                     gamma=0.99,
                     q_model_config={"type": FQETorchModel, "n_iters": 10, "lr": 0.0005})              # ajusta learning‑rate si quieres

print("\n⏳ Entrenando Q‑model (DM)...")
for i in range(50):
    r = dm_est.train(reader_train.next())
    if (i + 1) % 10 == 0:
        print(f" iter {i+1:>2}: loss={r['loss']:.4f}")

# --------------------------------------------------------------------------- #
# 4. INSTANCIAR IS y DR (DR usa Q‑model + IS)
# --------------------------------------------------------------------------- #
is_est = ImportanceSampling(policy=eval_agent, gamma=0.99)
dr_est = DoublyRobust(policy=eval_agent,
                     q_estimator=dm_est,    # reutiliza el Q‑model aprendido
                     gamma=0.99)

# (IS y DR no necesitan fase de 'train', pero si tuvieran parámetros, aquí iría.)

# --------------------------------------------------------------------------- #
# 5. EVALUAR π_target SOBRE EL DATASET DE TEST
# --------------------------------------------------------------------------- #
reader_test = JsonReader(TEST_JSON)
batch_test  = reader_test.next()

def n_eps(batch): return int(np.sum(batch["dones"]))

print(f"\n📊 Batch de test: {len(batch_test['rewards'])} transiciones, {n_eps(batch_test)} episodios")

dm_value = dm_est.estimate(batch_test)
is_value = is_est.estimate(batch_test)
dr_value = dr_est.estimate(batch_test)

print("\n=======  RESULTADOS OPE  =======")
print(f"Direct Method (DM) : {dm_value:.3f}")
print(f"ImportanceSampling : {is_value:.3f}")
print(f"Doubly Robust (DR) : {dr_value:.3f}")