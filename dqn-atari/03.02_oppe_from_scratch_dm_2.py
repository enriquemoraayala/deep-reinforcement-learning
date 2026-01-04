"""
Off‑Policy Policy Evaluation con RLlib 2.11
==========================================

"""
from __future__ import annotations
import numpy as np, gymnasium as gym
import debugpy
import pandas as pd
import torch
import torch.nn as nn
import ray
import json

from oppe_utils import load_checkpoint, load_json_to_df_max, calculate_policy_expected_value, load_fqte
from statistics import mean, stdev
from ray.rllib.policy.policy import Policy
from ray.rllib.offline.json_reader import JsonReader
from typing import Tuple



ray.init(ignore_reinit_error=True, include_dashboard=False)
print(ray.__version__)

debug = 1
generate_eps = 0

if debug == 1:
    # Escucha en el puerto 5678 (puedes cambiarlo)
    debugpy.listen(("0.0.0.0", 5678))
    print("Esperando debugger de VS Code para conectar...")
    debugpy.wait_for_client()


def extract_initial_observations(
    df: pd.DataFrame,
    obs_column: str = "obs",
    episode_id_column: str = "ep",
    timestep_column: str = "step",
) -> np.ndarray:
    """
    Extrae el estado inicial s0 de cada episodio a partir del DataFrame.

    Requisitos:
      - df tiene una fila por transición.
      - `episode_id` identifica el episodio.
      - `timestep` indica el orden temporal (0,1,2,...).
      - `obs` contiene la observación s_t (np.array de forma [obs_dim]).

    Devuelve un array de shape [N_episodios, obs_dim].
    """
    # Ordenamos para que timestep=0 esté correctamente identificado
    df_sorted = df.sort_values([episode_id_column, timestep_column])

    # Tomamos la primera fila de cada episodio
    first_rows = df_sorted.groupby(episode_id_column).head(1)

    # Convertimos la columna de observaciones a un np.array
    obs_list = first_rows[obs_column].to_list()
    s0_batch = np.stack(obs_list, axis=0)  # [N_episodios, obs_dim]

    return s0_batch



# ==========================
# 4. Obtener π_e(a|s) desde la política PPO de RLlib
# ==========================

def policy_action_probs_from_rllib(
    policy: Policy,
    obs_batch: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Calcula las probabilidades de acción π_e(a|s) para un batch de observaciones.

    Para políticas discretas de RLlib (PPO + Torch):
      - policy.model(input_dict)["obs"] -> logits de acciones.

    Nota: La API interna de RLlib puede variar ligeramente entre versiones.
    """
    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
    input_dict = {"obs": obs_tensor}
    model_out, _ = policy.model(input_dict, [], None)  # type: ignore
    logits = model_out  # [batch_size, num_actions]

    probs = torch.softmax(logits, dim=-1)  # π_e(a|s)
    return probs




#################OPE Methods###################

# ==========================
# 5. DM usando FQE + π_e
# ==========================

def dm_estimate(
    fqe_model: nn.Module,
    policy: Policy,
    s0_batch: np.ndarray,
    device: torch.device,
    use_stochastic_policy: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Estimación Direct Method del valor de π_e usando FQE.

    Parámetros:
      - fqe_model: red que aproxima Q^{π_e}(s,a).
      - policy: política PPO de RLlib.
      - s0_batch: np.ndarray [N_episodios, obs_dim], estados iniciales.
      - device: torch.device.
      - use_stochastic_policy:
          True  -> usa E_{a~π_e}[Q(s,a)].
          False -> usa max_a Q(s,a) (política greedy w.r.t Q).

    Devuelve:
      - j_hat: escalar con la estimación del retorno medio.
      - v_values: np.ndarray [N_episodios] con V(s0) episodio a episodio.
    """
    fqe_model.eval()

    with torch.no_grad():
        obs_tensor = torch.tensor(s0_batch, dtype=torch.float32, device=device)

        # Q(s, a) para todas las acciones
        q_values = fqe_model(obs_tensor)  # [N, num_actions]

        if use_stochastic_policy:
            # π_e(a|s)
            probs = policy_action_probs_from_rllib(policy, s0_batch, device)  # [N, num_actions]
            # V(s) = sum_a π(a|s) Q(s,a)
            v_tensor = (probs * q_values).sum(dim=1)  # [N]
        else:
            # Política greedy w.r.t Q
            v_tensor, _ = q_values.max(dim=1)  # [N]

        v_values = v_tensor.cpu().numpy()
        j_hat = float(v_values.mean())

    return j_hat, v_values




def oppe():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    BEH_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/120820251600"
    EVAL_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/130820251600"
    FQE_CHECKPOINT_PATH = "./fqe_checkpoints"
    
    BEH_EPISODES_JSON_TRAIN = '/opt/ml/code/episodes/120820251600/011125_01_generated_rllib_ppo_rllib_seed_0000_10000eps_300steps_exp_0'
    BEH_EPISODES_JSON_TEST = '/opt/ml/code/episodes/120820251600/011125_generated_rllib_ppo_rllib_seed_0000_2000eps_300steps_exp_0'
    BEH_EPISODES_JSON = '/opt/ml/code/episodes/120820251600/011125_generated_rllib_ppo_rllib_seed_0000_1000eps_300steps_exp_0'
    EVAL_EPISODES_JSON = '/opt/ml/code/episodes/130820251600/140825_generated_rllib_ppo_rllib_seed_0000_1000eps_200steps_exp_0'
    
    # beh_policy = load_checkpoint(BEH_CHECKPOINT_PATH)
    eval_policy = load_checkpoint(EVAL_CHECKPOINT_PATH)

    # ---------------------------------------------------------------------------#
    # LEYENDO DATOS
    # --------------------------------------------------------------------------- #
    reader_beh = JsonReader(BEH_EPISODES_JSON_TEST)
    reader_beh_train = JsonReader(BEH_EPISODES_JSON_TRAIN)
    reader_beh_test = JsonReader(BEH_EPISODES_JSON_TEST)
    reader_target = JsonReader(EVAL_EPISODES_JSON)
    beh_eps_df = load_json_to_df_max(reader_beh, 2000)
    target_eps_df = load_json_to_df_max(reader_target, 1000)
    beh_expected_return, beh_return_stdev = calculate_policy_expected_value(beh_eps_df, 0.99)
    target_expected_return, target_return_stdev = calculate_policy_expected_value(target_eps_df, 0.99)
    print(f"Avg_Expecting_Return (BEH_POLICY) Value - RLLIB Generated episodes: {beh_expected_return: .3f} - STD {beh_return_stdev: .3f}")
    print(f"Avg_Expecting_Return (TARGET_POLICY) Value - RLLIB Generated episodes: {target_expected_return: .3f} - STD {target_return_stdev: .3f}")

    print("Extrayendo estados iniciales s0 de cada episodio...")
    s0_batch = extract_initial_observations(
        beh_eps_df,
        obs_column="obs",             
        episode_id_column="ep",
        timestep_column="step",
    )
    print(f"Número de episodios: {s0_batch.shape[0]}")

    # 3. Cargar FQE
    print("Cargando modelo FQE...")
    fqe_model = load_fqte(FQE_CHECKPOINT_PATH, device)

    # 4. Estimación DM con política estocástica
    print("Calculando estimador DM (política estocástica)...")
    j_dm_stochastic, v_values_stochastic = dm_estimate(
        fqe_model=fqe_model,
        policy=eval_policy,
        s0_batch=s0_batch,
        device=device,
        use_stochastic_policy=True,
    )
    print(f"[DM estocástico] J_hat ≈ {j_dm_stochastic:.4f}")

    # 6. Estimación DM con política greedy respecto a Q
    print("Calculando estimador DM (política greedy w.r.t Q)...")
    j_dm_greedy, v_values_greedy = dm_estimate(
        fqe_model=fqe_model,
        policy=eval_policy,
        s0_batch=s0_batch,
        device=device,
        use_stochastic_policy=False,
    )
    print(f"[DM greedy] J_hat ≈ {j_dm_greedy:.4f}")
    ray.shutdown()

  
if __name__ == '__main__':
    oppe()
