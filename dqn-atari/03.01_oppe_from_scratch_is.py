"""
Off‑Policy Policy Evaluation con RLlib 2.11
==========================================

"""
from __future__ import annotations
import os, json, random, numpy as np, gymnasium as gym
import debugpy
import pandas as pd
import torch
import ray

from oppe_utils import load_checkpoint, load_json_to_df_max, calculate_policy_expected_value
from oppe_utils import add_target_logprobs_from_rllib, add_target_probs, add_target_probs_log
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

debug = 0
generate_eps = 0

if debug == 1:
    # Escucha en el puerto 5678 (puedes cambiarlo)
    debugpy.listen(("0.0.0.0", 5678))
    print("Esperando debugger de VS Code para conectar...")
    debugpy.wait_for_client()

#################OPE Methods###################


def ordinary_is_ope(df, gamma: float = 0.99, eps: float = 1e-8):
    df = df.sort_values(["ep", "step"]).copy()
    
    # Evitar divisiones por cero
    df["prob_accion_clipped"] = df["action_prob"].clip(lower=eps)
    df["target_prob_accion_clipped"] = df["target_prob_accion"].clip(lower=eps)
    
    # Ratios por paso
    df["rho_t"] = df["target_prob_accion_clipped"] / df["prob_accion_clipped"]
    
    def episode_is_return(group):
        rewards = group["reward"].to_numpy()
        rhos = group["rho_t"].to_numpy()
        
        T = len(group)
        discounts = gamma ** np.arange(T)
        
        # Retorno del episodio bajo la behavior (solo con rewards)
        G = np.sum(discounts * rewards)
        
        # Peso de importance sampling (producto de rhos)
        w = np.prod(rhos)
        
        return w * G, w, G
    
    # Aplicamos por episodio
    results = df.groupby("ep").apply(
        lambda g: pd.Series(episode_is_return(g), index=["wG", "w", "G"])
    )
    
    # Ordinary IS estimator
    V_IS = results["wG"].mean()
    
    return {
        "V_IS": V_IS,
        "episodic_table": results,  # por si quieres inspeccionar
    }


def ordinary_is_ope_log(df, gamma: float = 0.99, max_log_w_clip: float | None = 20.0):
    """
    Off-policy evaluation con Ordinary Importance Sampling usando log-probs.
    
    Requiere columnas:
    - 'ep'
    - 'step'
    - 'reward'
    - 'logprob'          (log π_b(a|s))
    - 'target_logprob_accion'   (log π_e(a|s))
    """
    df = df.sort_values(["ep", "step"]).copy()
    
    # log-ratio por paso
    df["log_rho_t"] = df["target_logprob_action"] - df["logprob"]
    
    def episode_is_return_log(group):
        rewards = group["reward"].to_numpy()
        log_rhos = group["log_rho_t"].to_numpy()
        
        T = len(group)
        discounts = gamma ** np.arange(T)
        
        # Retorno del episodio (solo rewards)
        G = np.sum(discounts * rewards)
        
        # log peso de importance sampling
        log_w = np.sum(log_rhos)
        
        # Clipping opcional para estabilidad numérica
        if max_log_w_clip is not None:
            log_w = np.clip(log_w, -max_log_w_clip, max_log_w_clip)
        
        w = np.exp(log_w)
        
        return w * G, w, G, log_w
    
    results = df.groupby("ep").apply(
        lambda g: pd.Series(episode_is_return_log(g), index=["wG", "w", "G", "log_w"])
    )
    
    V_IS = results["wG"].mean()
    
    return {
        "V_IS": V_IS,
        "episodic_table": results,  # incluye pesos, retornos y log_w por episodio
    }


def weighted_is_ope_log(df, gamma: float = 0.99, max_log_w_clip: float | None = 20.0):
    """
    Weighted Importance Sampling con log-probs.
    """
    df = df.sort_values(["ep", "step"]).copy()
    df["log_rho_t"] = df["target_logprob_action"] - df["logprob"]
    
    def episode_is_return_log(group):
        rewards = group["reward"].to_numpy()
        log_rhos = group["log_rho_t"].to_numpy()
        
        T = len(group)
        discounts = gamma ** np.arange(T)
        G = np.sum(discounts * rewards)
        
        log_w = np.sum(log_rhos)
        if max_log_w_clip is not None:
            log_w = np.clip(log_w, -max_log_w_clip, max_log_w_clip)
        w = np.exp(log_w)
        
        return w * G, w, G, log_w
    
    results = df.groupby("ep").apply(
        lambda g: pd.Series(episode_is_return_log(g), index=["wG", "w", "G", "log_w"])
    )
    
    num = results["wG"].sum()
    den = results["w"].sum()
    
    V_WIS = num / den if den != 0.0 else np.nan
    
    return {
        "V_WIS": V_WIS,
        "episodic_table": results,
    }


def oppe():

    BEH_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/120820251600"
    EVAL_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/130820251600"
    FQE_CHECKPOINT_PATH = "./fqe_checkpoints"
    
    BEH_EPISODES_JSON_TRAIN = '/opt/ml/code/episodes/120820251600/011125_01_generated_rllib_ppo_rllib_seed_0000_10000eps_300steps_exp_0'
    BEH_EPISODES_JSON_TEST = '/opt/ml/code/episodes/120820251600/011125_generated_rllib_ppo_rllib_seed_0000_2000eps_300steps_exp_0'
    BEH_EPISODES_JSON_VAL = '/opt/ml/code/episodes/120820251600/011125_generated_rllib_ppo_rllib_seed_0000_1000eps_300steps_exp_0'
    EVAL_EPISODES_JSON = '/opt/ml/code/episodes/130820251600/011125_generated_rllib_ppo_rllib_seed_0000_1000eps_300steps_exp_0'
    
    # beh_policy = load_checkpoint(BEH_CHECKPOINT_PATH)
    eval_policy = load_checkpoint(EVAL_CHECKPOINT_PATH)

    # ---------------------------------------------------------------------------#
    # LEYENDO DATOS
    # --------------------------------------------------------------------------- #
    reader_beh_val = JsonReader(BEH_EPISODES_JSON_VAL)
    reader_beh_train = JsonReader(BEH_EPISODES_JSON_TRAIN)
    reader_beh_test = JsonReader(BEH_EPISODES_JSON_TEST)
    reader_target = JsonReader(EVAL_EPISODES_JSON)
    beh_eps_d_val, eps, steps = load_json_to_df_max(reader_beh_val, 100000)
    print(f'loaded JSON: {BEH_EPISODES_JSON_VAL}')
    print(f"Transformed {eps} episodes with a total of {steps} steps")
    target_eps_df, eps, steps = load_json_to_df_max(reader_target, 100000)
    print(f'loaded JSON: {EVAL_EPISODES_JSON}')
    print(f"Transformed {eps} episodes with a total of {steps} steps")
    beh_expected_return, beh_return_stdev = calculate_policy_expected_value(beh_eps_d_val, 0.99)
    target_expected_return, target_return_stdev = calculate_policy_expected_value(target_eps_df, 0.99)
    print(f"Avg_Expecting_Return (BEH_POLICY) Value - RLLIB Generated episodes: {beh_expected_return: .3f} - STD {beh_return_stdev: .3f}")
    print(f"Avg_Expecting_Return (TARGET_POLICY) Value - RLLIB Generated episodes: {target_expected_return: .3f} - STD {target_return_stdev: .3f}")

    print("Calculating IS, WIS from scratch")
    beh_eps_df_test, eps, steps = load_json_to_df_max(reader_beh_test, 100000)
    print(f'loaded JSON: {BEH_EPISODES_JSON_TEST}')
    print(f"Transformed {eps} episodes with a total of {steps} steps")

    df = add_target_probs(beh_eps_df_test, eval_policy)
    df_log = add_target_probs_log(beh_eps_df_test, eval_policy)
    df_with_target = add_target_logprobs_from_rllib(beh_eps_df_test, eval_policy)

    res_is   = ordinary_is_ope(df, gamma=0.99)
    res_is_log   = ordinary_is_ope_log(df_log, gamma=0.99)
    res_is_log_vect   = ordinary_is_ope_log(df_with_target, gamma=0.99)
    
    res_wis_log  = weighted_is_ope_log(df_log, gamma=0.99)
    res_wis_log_vect  = weighted_is_ope_log(df_with_target, gamma=0.99)
    # res_pdis = per_decision_is_ope(df, gamma=0.99)

    print("Ordinary IS:", res_is["V_IS"])
    print("Ordinary IS (logprob):", res_is_log["V_IS"])
    print("Ordinary IS (vect logprob):", res_is_log_vect["V_IS"])
    
    print("Weighted IS (logp):", res_wis_log["V_WIS"])
    print("Weighted IS (vect logp):", res_wis_log_vect["V_WIS"])
    # print("Per-decision IS:", res_pdis["V_PDIS"])
   
    ray.shutdown()

  
if __name__ == '__main__':
    oppe()
