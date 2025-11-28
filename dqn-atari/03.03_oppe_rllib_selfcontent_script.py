"""
Off‑Policy Policy Evaluation con RLlib 2.11
==========================================

• Se entrena (rápidamente) una PPO ‑> será la **política objetivo** (π_target).
• Se crea una *behavior policy* distinta y se generan episodios offline (.json).
• Se entrena un Q‑model para los estimadores que lo necesitan (DM y DR).
• Se instancian y entrenan DM, IS y DR con los datos offline.
• Se estiman los retornos de π_target aprovechando los estimadores entrenados.
"""
from __future__ import annotations
import os, json, random, numpy as np, gymnasium as gym
import debugpy
import pandas as pd
import torch
import ray

from oppe_utils import load_checkpoint, load_json_to_df, calculate_policy_expected_value
from oppe_utils import QNetwork
from statistics import mean, stdev
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.offline.json_reader import JsonReader

from ray.rllib.algorithms import Algorithm
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.offline.estimators.fqe_torch_model import FQETorchModel

import argparse, ast, json
from pathlib import Path
from typing import List, Dict, Any

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.sample_batch import SampleBatch

# Estimadores nativos RLlib 2.x
from ray.rllib.offline.estimators import (
    ImportanceSampling as RL_IS,
    WeightedImportanceSampling as RL_WIS,
    DirectMethod as RL_DM,
    DoublyRobust as RL_DR,
)

ray.init(ignore_reinit_error=True, include_dashboard=False)
print(ray.__version__)

debug = 1
generate_eps = 0

if debug == 1:
    # Escucha en el puerto 5678 (puedes cambiarlo)
    debugpy.listen(("0.0.0.0", 5678))
    print("Esperando debugger de VS Code para conectar...")
    debugpy.wait_for_client()


# ---------------- Utilidades ----------------
def _to_array(x):
    if isinstance(x, (list, np.ndarray)):
        arr = np.asarray(x, dtype=np.float32)
    elif isinstance(x, str):
        try:
            arr = np.asarray(json.loads(x), dtype=np.float32)
        except Exception:
            arr = np.asarray(ast.literal_eval(x), dtype=np.float32)
    else:
        arr = np.asarray(x, dtype=np.float32)
    return arr


def _group_episodes(df, ep_col, step_col):
    df = df.sort_values([ep_col, step_col]).reset_index(drop=True)
    for ep_id, g in df.groupby(ep_col, sort=True):
        yield ep_id, g


# Conversión DataFrame -> lista de episodios (dict) para estimadores RLlib


def pandas_to_episodes(
    df: pd.DataFrame,
    *, ep_col="ep", step_col="step", obs_col="obs", next_obs_col="next_obs",
    action_col="action", reward_col="reward", term_col="terminated",
    beh_prob_col="action_prob", beh_logp_col="action_logp",
    ) -> List[Dict[str, Any]]:
    episodes = []
    max_steps = 200 # definido desde el principio
    for _, g in _group_episodes(df, ep_col, step_col):
        obs = np.vstack(g[obs_col].map(_to_array).to_list())
        next_obs = np.vstack(g[next_obs_col].map(_to_array).to_list()) if next_obs_col in g else None
        acts = g[action_col].astype(np.int64).to_numpy()
        rews = g[reward_col].astype(np.float32).to_numpy()
        terms = g[term_col].astype(bool).to_numpy() if term_col in g else np.zeros(len(g), dtype=bool)
        # episodios truncados, no lo habia guardado en el dataset
        # Inicializa todos los pasos como no truncados
        truncs = np.zeros_like(terms, dtype=bool)
        # Si el episodio no terminó naturalmente y alcanzó el límite
        if not terms.any() and len(terms) == max_steps:
            truncs[-1] = True  # el último paso se marca como trunc
        epi = {
        "obs": obs,
        "next_observations": next_obs,
        "actions": acts,
        "rewards": rews,
        "terminateds": terms,
        "trunctateds": truncs,
        }
        if beh_prob_col in g:
            epi["action_prob"] = g[beh_prob_col].astype(np.float32).to_numpy()
        elif beh_logp_col in g:
            epi["action_prob"] = np.exp(g[beh_logp_col].astype(np.float32).to_numpy())
        episodes.append(epi)
    return episodes

# ---------------- OPE con estimadores RLlib ----------------

def define_rllib_estimators(algo):
    dr_estimator = RL_DR(
        # policy=algo.get_policy(),
        policy=algo,
        gamma=0.99,
        q_model_config={"type": FQETorchModel, "n_iters": 100, "lr": 0.0005, "polyak_coef": 0.05},
        )

    is_estimator = RL_IS(
        policy=algo,
        gamma=0.99,
        epsilon_greedy=0.05
    )

    wis_estimator = RL_WIS(
        policy=algo,
        gamma=0.99,
        epsilon_greedy=0.05
    )

    dm_estimator = RL_DM(
        policy=algo,
        gamma=0.99,
        q_model_config={"type": FQETorchModel, "n_iters": 100, "lr": 0.0005, "polyak_coef": 0.05},
    )

    return dr_estimator, is_estimator, dm_estimator


def evaluate_with_rllib_estimators(
    episodes: List[Dict[str, Any]],
    eval_algo: Algorithm,
    *,
    gamma: float = 0.99,
    ) -> Dict[str, Any]:
    policy = eval_algo
    est_classes = [RL_IS, RL_WIS, RL_DM, RL_DR]
    results = {}
    for cls in est_classes:
        est = cls(policy=policy, gamma=gamma)
        res = est.estimate(episodes)
        results[cls.__name__] = res
    return results


def train_rllib_dm_dr_models(BEH_EPISODES_JSON_TRAIN, reader_beh_train, dr_estimator, dm_estimator):
    print("\n⏳ Entrenando Q‑model DM and DR ...")
    for i in range(10000):
        batch = reader_beh_train.next()
        dm_r = dm_estimator.train(batch)
        dr_r = dr_estimator.train(batch)
        if (i + 1) % 200 == 0:
            print(f" DM: iter {i+1:>2}: loss={dm_r['loss']:.4f}")
            print(f" DR: iter {i+1:>2}: loss={dr_r['loss']:.4f}")
  
    return dm_estimator, dr_estimator


def oppe():

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
    reader_beh = JsonReader(BEH_EPISODES_JSON)
    reader_beh_train = JsonReader(BEH_EPISODES_JSON_TRAIN)
    reader_beh_test = JsonReader(BEH_EPISODES_JSON_TEST)
    reader_target = JsonReader(EVAL_EPISODES_JSON)
    beh_eps_df = load_json_to_df(reader_beh, 1000)
    target_eps_df = load_json_to_df(reader_target, 1000)
    beh_expected_return, beh_return_stdev = calculate_policy_expected_value(beh_eps_df, 0.99)
    target_expected_return, target_return_stdev = calculate_policy_expected_value(target_eps_df, 0.99)
    print(f"Avg_Expecting_Return (BEH_POLICY) Value - RLLIB Generated episodes: {beh_expected_return: .3f} - STD {beh_return_stdev: .3f}")
    print(f"Avg_Expecting_Return (TARGET_POLICY) Value - RLLIB Generated episodes: {target_expected_return: .3f} - STD {target_return_stdev: .3f}")


    # --------------------------------------------------------------------------- #
    # RLLIB OPPE: 
    # 
    # Vamos a cambiar, en lugar de usar el JSonReader, que no tengo control del batch
    # size, voy a intentar preparar el dataset para rllib a mano
    # --------------------------------------------------------------------------- #
    # df_columns = { 'ep_col': 'ep', 'step_col': 'step', 'obs_col': 'obs', 'next_obs_col': 'next_state', 
    #                'action_col': 'action', 'reward_col': 'reward', 'term_col': 'done', 
    #                'beh_prob_col': 'action_prob', 'beh_logp_col': 'logprob'}
    # episodes = pandas_to_episodes(
    #             beh_eps_df,
    #             ep_col=df_columns['ep_col'], step_col=df_columns['step_col'],
    #             obs_col=df_columns['obs_col'], next_obs_col=df_columns['next_obs_col'],
    #             action_col=df_columns['action_col'], reward_col=df_columns['reward_col'],
    #             term_col=df_columns['term_col'],
    #             beh_prob_col=df_columns['beh_prob_col'], beh_logp_col=df_columns['beh_logp_col'],
    #             )

    # # Ejecuta OPE con RLlib
    # results = evaluate_with_rllib_estimators(episodes, eval_policy, gamma=0.99)

    # print("=== Off-Policy Evaluation (RLlib estimators) ===")
    # for name, res in results.items():
    #     print(f"[{name}] -> {res}")


    dr_estimator, is_estimator, dm_estimator = define_rllib_estimators(eval_policy)
    reader_beh_test = JsonReader(BEH_EPISODES_JSON_TEST)
    is_values = []
    for idx in range(2000):
        batch_test  = reader_beh_test.next()
        is_value = is_estimator.estimate(batch_test)
        is_values.append(is_value['v_target'])
    print(f"Ordinary Importance Sampling (IS RLLIB) Value: {mean(is_values)}")

    dm_estimator, dr_estimator = train_rllib_dm_dr_models(BEH_EPISODES_JSON_TRAIN, reader_beh_train, dr_estimator, dm_estimator)
    dr_ests = []
    dm_ests = []
    for i in range(2000):
            batch = reader_beh_test.next()
            dr_ests.append(dr_estimator.estimate(batch)['v_target'])
            dm_ests.append(dm_estimator.estimate(batch)['v_target'])

    print(f'Direct Method (DM) with RLLIB V_ expected estimation {mean(dm_ests)} and STD {stdev(dm_ests)}')
    print(f'Double Roboust (DR) with RLLIB V_ expected estimation {mean(dr_ests)} and STD {stdev(dr_ests)}')

    ray.shutdown()

  
if __name__ == '__main__':
    oppe()
