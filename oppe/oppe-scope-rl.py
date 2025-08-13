import numpy as np
import pandas as pd
import gymnasium as gym
import torch, os
import debugpy

# Imports de Ray RLlib para cargar la política entrenada
import ray
from ray.rllib.algorithms import Algorithm

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.offline.json_reader import JsonReader

# Imports de Scope-RL para OPE
from scope_rl.ope import CreateOPEInput
from scope_rl.ope import OffPolicyEvaluation as OPE
from scope_rl.ope.discrete import DirectMethod as DM, TrajectoryWiseImportanceSampling as TIS
from scope_rl.ope.discrete import PerDecisionImportanceSampling as PDIS, DoublyRobust as DR

# (También podríamos importar DirectMethod (DM) y DoublyRobust (DR) si se van a usar)

from oppe_utils import load_checkpoint, load_json_to_df, calculate_value_function, calculate_return, QNetwork



# Envolver la política PPO en una clase con interfaz esperada
class PPOPolicyWrapperScopeRL:
    def __init__(self, ppo_policy, name="PPO_policy"):
        self.trainer = ppo_policy
        self.name = name
    def predict(self, state):
        # Devuelve la acción que tomaría la política (determinísticamente)
        action = self.trainer.compute_single_action(state, explore=False)[0]
        return action
    def predict_probabilities(self, state):
        # Devuelve las probabilidades de cada acción bajo la política PPO
        s_tensor = torch.tensor(state, dtype=torch.float32)
        logits, _ = self.ppo_policy.model({"obs": s_tensor}, [], None)
        prob_vec = torch.softmax(logits, dim=1)[0].detach().numpy()
        return prob_vec



debug = 1
generate_eps = 0

if debug == 1:
    # Escucha en el puerto 5678 (puedes cambiarlo)
    debugpy.listen(("0.0.0.0", 5678))
    print("Esperando debugger de VS Code para conectar...")
    debugpy.wait_for_client()

# Iniciar Ray (modo local)
ray.init(ignore_reinit_error=True)
print(ray.__version__)

BEH_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/120820251600"
# EVAL_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/300720251000"
EVAL_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/120820251600"
FQE_CHECKPOINT_PATH = "./fqe_checkpoints"

BEH_EPISODES_JSON_TRAIN = '/opt/ml/code/episodes/310720251600/310725_generated_rllib_ppo_rllib_seed_0000_2000eps_200steps_exp_0'
BEH_EPISODES_JSON_TEST = '/opt/ml/code/episodes/310720251600/310725_generated_rllib_ppo_rllib_seed_0000_2000eps_200steps_exp_8'
BEH_EPISODES_JSON = '/opt/ml/code/episodes/310720251600/010825_generated_rllib_ppo_rllib_seed_0000_1000eps_200steps_exp_0'
EVAL_EPISODES_JSON = '/opt/ml/code/episodes/300720251000/050825_generated_rllib_ppo_rllib_seed_0000_1000eps_200steps_exp_0'
# Restaurar el agente PPO entrenado desde el checkpoint


eval_agent = Algorithm.from_checkpoint(EVAL_CHECKPOINT_PATH)
target_policy = eval_agent.get_policy()  # Obtenemos la política (por defecto "default_policy")

# Cargar el dataset offline en un DataFrame (ya disponible como beh_df, o se lee de un archivo)
# Suponemos que beh_df tiene columnas: 'ep', 'obs', 'action', 'reward', 'done', 'action_prob'
# Si los datos vienen de JSON, asegúrese de convertir listas a numpy arrays apropiadamente.

# ---------------------------------------------------------------------------#
# LEYENDO DATOS
# --------------------------------------------------------------------------- #
reader_beh = JsonReader(BEH_EPISODES_JSON)
reader_beh_train = JsonReader(BEH_EPISODES_JSON_TRAIN)
reader_beh_test = JsonReader(BEH_EPISODES_JSON_TEST)
reader_target = JsonReader(EVAL_EPISODES_JSON)
n_trajectories_target = 2000
n_trajectories_beh = 2000
beh_eps_df = load_json_to_df(reader_beh, n_trajectories_beh)
beh_test_df = load_json_to_df(reader_beh_test, n_trajectories_beh)
target_eps_df = load_json_to_df(reader_target, n_trajectories_target)
beh_expected_return, beh_return_stdev = calculate_value_function(beh_eps_df, 0.99)
target_expected_return, target_return_stdev = calculate_value_function(target_eps_df, 0.99)
print(f"Avg_Expecting_Return (BEH_POLICY) Value - RLLIB Generated episodes: {beh_expected_return: .3f} - STD {beh_return_stdev: .3f}")
print(f"Avg_Expecting_Return (TARGET_POLICY) Value - RLLIB Generated episodes: {target_expected_return: .3f} - STD {target_return_stdev: .3f}")


# Determinar dimensiones y parámetros del entorno
env = gym.make("LunarLander-v3")
state_dim = 8
n_actions = env.action_space.n  # número de acciones (LunarLander-v2 tiene 4)
try:
    max_steps = env.spec.max_episode_steps
except:
    max_steps = 200  # usar 200 pasos máximo conforme se limitó en la generación de datos
max_steps = 200
env.close()

n_episodes = beh_eps_df['ep'].nunique()
gamma = 0.99  # factor de descuento usado para OPE

# Inicializar arrays vacíos para el dataset completo con padding
# en nuestro caso siempre son de 200 pero dejo el codigo por si acaso lo hacemos variable
# en el futuro
total_steps = n_trajectories_target * max_steps
state_arr = np.zeros((total_steps, state_dim), dtype=np.float32)
action_arr = np.zeros(total_steps, dtype=np.int32)
reward_arr = np.zeros(total_steps, dtype=np.float32)
done_arr = np.zeros(total_steps, dtype=bool)
terminal_arr = np.zeros(total_steps, dtype=bool)
pscore_arr = np.zeros(total_steps, dtype=np.float32)

# Rellenar episodio por episodio
current = 0
for ep, ep_data in beh_eps_df.groupby("ep"):
    ep_len = len(ep_data)
    # Copiar datos reales del episodio
    state_arr[current: current+ep_len]   = np.stack(ep_data["obs"].values)
    action_arr[current: current+ep_len]  = ep_data["action"].to_numpy()
    reward_arr[current: current+ep_len]  = ep_data["reward"].to_numpy()
    pscore_arr[current: current+ep_len]  = ep_data["action_prob"].to_numpy()
    # Marcar el último paso del episodio
    done_arr[current + ep_len - 1] = True
    terminal_arr[current + ep_len - 1] = True
    # Avanzar al siguiente bloque de episodio (saltando padding restante)
    current += max_steps
# Nota: Los indices entre current+ep_len y current+max_steps quedan en cero (padding)


logged_dataset = {
    "size": int(total_steps),             # total de transiciones (incluyendo padding)
    "n_trajectories": int(n_trajectories_beh),
    "step_per_trajectory": int(max_steps),
    "state_dim": int(state_dim),
    "action_type": 'discrete',           # "discrete" para LunarLander-v3
    "n_actions": int(n_actions),
    "state": state_arr,
    "action": action_arr,
    "reward": reward_arr,
    "done": done_arr,
    "terminal": terminal_arr,
    "pscore": pscore_arr,
    "behavior_policy": None,              # podemos dejar None si no tenemos un objeto de política comportamiento
    "dataset_id": 0,
    "action_meaning": None
}

ppo_policy = PPOPolicyWrapperScopeRL(target_policy)

# Crear input para OPE combinando los datos offline con la política PPO
prep = CreateOPEInput(env=env)
input_dict = prep.obtain_whole_inputs(
    logged_dataset=logged_dataset,
    evaluation_policies=[ppo_policy],
    require_value_prediction=True,   # True para que se realice Fitted Q Evaluation (DM) para la política
    require_weight_prediction=False, # False porque ya tenemos pscore (propensiones) en logged_dataset
    n_trajectories_on_policy_evaluation=0,  # 0 si no queremos simular episodios on-policy para validación
    random_state=42
)

# Inicializar la evaluación off-policy con estimadores DM, TIS, PDIS y DR
ope = OPE(
    logged_dataset=logged_dataset,
    ope_estimators=[DM(), TIS(), PDIS(), DR()]
)

# Realizar la evaluación off-policy de la política PPO
ope_results = ope.visualize_off_policy_estimates(input_dict)

# (Opcional) Calcular manualmente los valores estimados por cada método
for estimator in ope.ope_estimators:
    est_name = estimator.estimator_name
    estimated_value = estimator.estimate_policy_value(input_dict["PPO_policy"])
    print(f"{est_name}: {estimated_value:.2f}")
