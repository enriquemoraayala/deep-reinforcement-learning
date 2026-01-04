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
import pandas as pd
import torch

from oppe_utils import load_checkpoint, load_json_to_df_max, calculate_return, calculate_policy_expected_value
from oppe_utils import QNetwork
from statistics import mean, stdev
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.offline.json_reader import JsonReader

import ray
ray.init(ignore_reinit_error=True, include_dashboard=False)
print(ray.__version__)

debug = 0
generate_eps = 0

if debug == 1:
    # Escucha en el puerto 5678 (puedes cambiarlo)
    debugpy.listen(("0.0.0.0", 5678))
    print("Esperando debugger de VS Code para conectar...")
    debugpy.wait_for_client()


def evaluate_policy_dm(q_net, df, policy_action):

    states = torch.tensor(np.stack(df['obs'].values), dtype=torch.float32)

    # == Evaluación de la política PPO usando el modelo Q entrenado ==
    q_net.eval()  # ponemos la red en modo evaluación
    # Identificamos estados iniciales de cada episodio en el DataFrame.
    # Suponemos que el DataFrame está ordenado por episodios, y que 'done' indica fin de episodio.
    initial_state_indices = []
    N = len(df)
    for i in range(N):
        if i == 0 or df.iloc[i-1]['done']:  # es el primer paso de un episodio si es el inicio del df o el registro anterior tuvo done=True
            initial_state_indices.append(i)

    initial_states = states[initial_state_indices]
    # Calculamos el valor promedio de la política PPO desde los estados iniciales.
    values = []
    for s in initial_states:
        # Obtenemos la acción sugerida por la política PPO en el estado inicial
        a,_,info = policy_action.compute_single_action(s, explore=False)  # acción (entera) de la política PPO
        # Calculamos Q(s, a) con nuestra red entrenada
        q_value = q_net(s)  # hacemos forward con tamaño de batch 1
        q_value_sa = q_value[0, a].item()
        values.append(q_value_sa)
    
    estimated_value = np.mean(values)
    std = np.std(values)
    return values, estimated_value, std


def doubly_robust(df_episodes, target_policy, q_net, gamma=0.99):
    """Estimate target policy value via Doubly Robust estimator (per-decision DR)."""
    num_episodes = int(df_episodes["ep"].nunique())
    dr_estimates = []
    episode_returns = []
    for ep_id in range(num_episodes):
        ep_df = df_episodes[df_episodes["ep"] == ep_id]
        # Baseline from Q at initial state under target policy:
        s0 = torch.tensor(ep_df.iloc[0]["obs"], dtype=torch.float32)
        a0, _, _ = target_policy.compute_single_action(s0, explore=False)
        baseline = q_net(s0)[0, int(a0)].item()  # Q(s0, π(s0))
        # Iterate over steps for DR correction terms
        dr_value = baseline
        rho = 1.0
        for t, step in ep_df.iterrows():
            s = step["obs"]; a = int(step["action"]); r = step["reward"]; done = step["done"]
            behavior_prob = step["action_prob"]
            # Target policy probability for the action taken by behavior:
            # s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            s_tensor = torch.tensor(s, dtype=torch.float32)
            logits, _ = target_policy.model({"obs": s_tensor}, [], None)
            prob_vec = torch.softmax(logits, dim=1)[0].detach().numpy()
            target_prob = prob_vec[a]
            # Update cumulative importance weight
            if behavior_prob == 0:
                rho = 0.0
            else:
                rho *= target_prob / behavior_prob
            # Compute Q-values needed for DR
            # Q_sa = q_net(torch.tensor(s, dtype=torch.float32).unsqueeze(0))[0, a].item()      # Q(s_t, a_t)ç
            Q_sa = q_net(s_tensor)[0, a].item()      # Q(s_t, a_t)
            if done:
                Q_next = 0.0  # terminal state value
            else:
                s_next = torch.tensor(step["next_state"], dtype=torch.float32).unsqueeze(0)
                a_next, _, _ = target_policy.compute_single_action(s_next, explore=False)
                Q_next = q_net(s_next)[0, int(a_next)].item()                   # Q(s_{t+1}, π(s_{t+1}))
            # Add DR correction term for this step (discounted by gamma^t):
            dr_value += rho * (r + gamma * Q_next - Q_sa) * (gamma ** step["step"])
        # Record per-episode DR estimate and actual return
        dr_estimates.append(dr_value)
        episode_returns.append(calculate_return(ep_df, gamma))
    V_dr = float(np.mean(dr_estimates)) if dr_estimates else 0.0
    avg_behavior_return = mean(episode_returns) if episode_returns else 0.0
    std_behavior_return = np.std(episode_returns)
    return V_dr, avg_behavior_return, std_behavior_return


def oppe():

    BEH_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/120820251600"
    EVAL_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/130820251600"
    FQE_CHECKPOINT_PATH = "./fqe_checkpoints"
    
    BEH_EPISODES_JSON_TRAIN = '/opt/ml/code/episodes/120820251600/011125_01_generated_rllib_ppo_rllib_seed_0000_10000eps_300steps_exp_0'
    BEH_EPISODES_JSON_TEST = '/opt/ml/code/episodes/120820251600/011125_generated_rllib_ppo_rllib_seed_0000_2000eps_300steps_exp_0'
    BEH_EPISODES_JSON = '/opt/ml/code/episodes/120820251600/011125_generated_rllib_ppo_rllib_seed_0000_1000eps_300steps_exp_0'
    EVAL_EPISODES_JSON = '/opt/ml/code/episodes/130820251600/140825_generated_rllib_ppo_rllib_seed_0000_1000eps_200steps_exp_0'
    # EVAL_EPISODES_JSON = '/opt/ml/code/episodes/300720251000/100825_generated_rllib_ppo_rllib_seed_0000_50eps_200steps_exp_0'
    beh_agent = load_checkpoint(BEH_CHECKPOINT_PATH)
    eval_agent = load_checkpoint(EVAL_CHECKPOINT_PATH)
    #  if generate_eps == 1:
        # generate_offline_json(TRAIN_JSON, beh_agent, num_episodes=100)
        # generate_offline_json(TEST_JSON, beh_agent, num_episodes=200)

    # ---------------------------------------------------------------------------#
    # LEYENDO DATOS
    # --------------------------------------------------------------------------- #
    reader_beh = JsonReader(BEH_EPISODES_JSON)
    reader_beh_train = JsonReader(BEH_EPISODES_JSON_TRAIN)
    reader_beh_test = JsonReader(BEH_EPISODES_JSON_TEST)
    reader_target = JsonReader(EVAL_EPISODES_JSON)
    beh_eps_df = load_json_to_df_max(reader_beh, 1000)
    beh_test_df = load_json_to_df_max(reader_beh_test, 2000)
    target_eps_df = load_json_to_df_max(reader_target, 1000)
    beh_expected_return, beh_return_stdev = calculate_policy_expected_value(beh_eps_df, 0.99)
    target_expected_return, target_return_stdev = calculate_policy_expected_value(target_eps_df, 0.99)
    print(f"Avg_Expecting_Return (BEH_POLICY) Value - RLLIB Generated episodes: {beh_expected_return: .3f} - STD {beh_return_stdev: .3f}")
    print(f"Avg_Expecting_Return (TARGET_POLICY) Value - RLLIB Generated episodes: {target_expected_return: .3f} - STD {target_return_stdev: .3f}")


    # --------------------------------------------------------------------------- #
    # DM FROM SCRATCH CON FQTE MODEL YA ENTRENADO
    # --------------------------------------------------------------------------- #

    if os.path.exists(FQE_CHECKPOINT_PATH):
        checkpoint = torch.load(FQE_CHECKPOINT_PATH + '/fqe_epoch_80.pt')
        q_net = QNetwork(8, 4)
        q_net.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        avg_loss = checkpoint["avg_loss"]
        print(f"Se cargó el checkpoint desde {FQE_CHECKPOINT_PATH}, entrenadas {start_epoch} épocas con Avg. Loss {avg_loss}")

    qsa_values, dm_estimated_value, dm_std = evaluate_policy_dm(q_net, beh_test_df, eval_agent)
    print(f"\nValor esperado estimado DM de la política PPO (retorno promedio inicial): {dm_estimated_value:.3f} - STD: {dm_std:.3f}")


    V_dr, dr_estimated_value, dr_std = doubly_robust(beh_test_df, eval_agent, q_net)
    print(f"\nValor esperado estimado DR de la política PPO (retorno promedio inicial): {dr_estimated_value:.3f} - STD: {dr_std:.3f}")

   
if __name__ == '__main__':
    oppe()
