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
from scope_rl.ope.discrete.basic_estimators import TrajectoryWiseImportanceSampling as TIS
from scope_rl.ope.discrete.basic_estimators import PerDecisionImportanceSampling as PDIS
from scope_rl.ope.discrete.basic_estimators import DirectMethod, DoublyRobust
# (También podríamos importar DirectMethod (DM) y DoublyRobust (DR) si se van a usar)


from oppe_utils import load_checkpoint, load_json_to_df, calculate_value_function, calculate_return, QNetwork

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

BEH_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/310720251600"
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
beh_eps_df = load_json_to_df(reader_beh, 1000)
beh_test_df = load_json_to_df(reader_beh_test, 2000)
target_eps_df = load_json_to_df(reader_target, 1000)
beh_expected_return, beh_return_stdev = calculate_value_function(beh_eps_df, 0.99)
target_expected_return, target_return_stdev = calculate_value_function(target_eps_df, 0.99)
print(f"Avg_Expecting_Return (BEH_POLICY) Value - RLLIB Generated episodes: {beh_expected_return: .3f} - STD {beh_return_stdev: .3f}")
print(f"Avg_Expecting_Return (TARGET_POLICY) Value - RLLIB Generated episodes: {target_expected_return: .3f} - STD {target_return_stdev: .3f}")




# Asegurar que los datos estén ordenados por episodio y timestep para procesarlos correctamente
# beh_df = beh_eps_df.sort_values(['ep', /*'timestep' o 'step' si existe*/]).reset_index(drop=True)

# Determinar dimensiones y parámetros del entorno
env = gym.make("LunarLander-v3")
n_actions = env.action_space.n  # número de acciones (LunarLander-v2 tiene 4)
try:
    max_steps = env.spec.max_episode_steps
except:
    max_steps = 200  # usar 200 pasos máximo conforme se limitó en la generación de datos
max_steps = 200
env.close()

n_episodes = beh_eps_df['ep'].nunique()
gamma = 0.99  # factor de descuento usado para OPE

# Preparar arrays numpy para las secuencias, rellenando episodios más cortos con pasos "dummy"
# Inicializamos arrays con valores neutrales (recompensa 0, prob. 1 para evitar alterar ratios).
actions = np.zeros((n_episodes, max_steps), dtype=int)
rewards = np.zeros((n_episodes, max_steps), dtype=float)
behavior_pscore = np.ones((n_episodes, max_steps), dtype=float)  # pscore=1 para pasos padding
# Distribución de la política target: por defecto asignamos prob=1 a acción 0 en pasos padding
target_action_dist = np.zeros((n_episodes, max_steps, n_actions), dtype=float)
target_action_dist[:, :, 0] = 1.0

# Convertir estados a numpy array para calcular distribuciones de π_target
# (Stack de observaciones; cada 'obs' se supone vector de estado continuo de LunarLander de dimensión 8)
obs_list = beh_eps_df['obs'].tolist()
obs_arr = np.array(obs_list, dtype=np.float32)  # shape: (total_transitions, state_dim)

# Obtener la distribución de acciones de la política objetivo para cada estado del dataset
# Usamos el modelo de la policy de RLlib: sacamos logits y aplicamos softmax.
input_dict = {"obs": torch.tensor(obs_arr)}
logits, _ = target_policy.model(input_dict, [], None)  # obtener logits de la policy para cada estado
prob_tensor = torch.nn.Softmax(dim=1)(logits)
policy_action_dist_all = prob_tensor.detach().numpy()  # shape: (N_total_steps, n_actions)

# Rellenar los arrays episodio por episodio
offset = 0
for i, ep_id in enumerate(sorted(beh_eps_df['ep'].unique())):
    ep_data = beh_eps_df[beh_eps_df['ep'] == ep_id]
    ep_len = len(ep_data)
    # Extraer datos reales del episodio
    actions[i, :ep_len] = ep_data['action'].to_numpy()
    rewards[i, :ep_len] = ep_data['reward'].to_numpy()
    behavior_pscore[i, :ep_len] = ep_data['action_prob'].to_numpy()
    # Distribución de la policy objetivo para cada paso real
    policy_probs_ep = policy_action_dist_all[offset: offset+ep_len]  # shape (ep_len, n_actions)
    target_action_dist[i, :ep_len, :] = policy_probs_ep
    offset += ep_len
    # Las porciones [i, ep_len:] ya están con reward=0, pscore=1 y distribución [1,0,...,0] (padding)

# Convertir a forma 1D los arrays para pasarlos a Scope-RL
actions_flat = actions.flatten()
rewards_flat = rewards.flatten()
behavior_pscore_flat = behavior_pscore.flatten()
target_action_dist_flat = target_action_dist.reshape(n_episodes * max_steps, n_actions)

# Ahora aplicamos los estimadores de OPE de Scope-RL:
tis_estimator = TIS()    # Trajectory-wise IS (importancia por trayectoria)
pdis_estimator = PDIS()  # Per-decision IS (importancia paso a paso)

# Calcular el retorno esperado estimado por cada método
tis_value = tis_estimator.estimate_policy_value(
    step_per_trajectory=max_steps,
    action=actions_flat,
    reward=rewards_flat,
    pscore=behavior_pscore_flat,
    evaluation_policy_action_dist=target_action_dist_flat,
    gamma=gamma
)
pdis_value = pdis_estimator.estimate_policy_value(
    step_per_trajectory=max_steps,
    action=actions_flat,
    reward=rewards_flat,
    pscore=behavior_pscore_flat,
    # evaluation_policy_action_dist=target_action_dist_flat[np.arange(len(actions_flat)), actions_flat],
    evaluation_policy_action_dist=target_action_dist_flat,
    gamma=gamma
)

print(f"Estimación de retorno por Trajectory-wise IS: {tis_value:.3f}")
print(f"Estimación de retorno por Per-decision IS: {pdis_value:.3f}")


if os.path.exists(FQE_CHECKPOINT_PATH):
    checkpoint = torch.load(FQE_CHECKPOINT_PATH + '/fqe_epoch_80.pt')
    q_net = QNetwork(8, 4)
    q_net.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint["epoch"]
    avg_loss = checkpoint["avg_loss"]
    print(f"Se cargó el checkpoint desde {FQE_CHECKPOINT_PATH}, entrenadas {start_epoch} épocas con Avg. Loss {avg_loss}")

# Supongamos que 'q_model' es una función/objeto que dado (s, a) devuelve Q^π(s,a) estimado.
dm_estimator = DirectMethod()  # Scope-RL usaría este Q para evaluar
dr_estimator = DoublyRobust()

dm_value = dm_estimator.estimate_policy_value(
    step_per_trajectory=max_steps,
    action=actions_flat,
    reward=rewards_flat,
    pscore=behavior_pscore_flat,
    evaluation_policy_action_dist=target_action_dist_flat,
    gamma=gamma
)
dr_value = dr_estimator.estimate_policy_value(
    step_per_trajectory=max_steps,
    action=actions_flat,
    reward=rewards_flat,
    pscore=behavior_pscore_flat,
    evaluation_policy_action_dist=target_action_dist_flat,
    gamma=gamma
)

print(f"Estimación de retorno por DM: {dm_value:.3f}")
print(f"Estimación de retorno por DR: {dr_value:.3f}")