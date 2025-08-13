import numpy as np
import os, json, random, numpy as np, gymnasium as gym
import debugpy
import pandas as pd
import torch

from oppe_utils import load_checkpoint, load_json_to_df, calculate_value_function, calculate_return
from statistics import mean
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.offline.json_reader import JsonReader

from ray.rllib.algorithms import Algorithm
from scope_rl.ope import OffPolicyEvaluation as OPE
from scope_rl.ope.continuous import DirectMethod as DM
from scope_rl.ope.continuous import TrajectoryWiseImportanceSampling as TIS
from scope_rl.ope.continuous import PerDecisionImportanceSampling as PDIS
from scope_rl.ope.continuous import DoublyRobust as DR

import ray
ray.init(ignore_reinit_error=True, include_dashboard=False)
print(ray.__version__)

debug = 1

if debug == 1:
    # Escucha en el puerto 5678 (puedes cambiarlo)
    debugpy.listen(("0.0.0.0", 5678))
    print("Esperando debugger de VS Code para conectar...")
    debugpy.wait_for_client()

BEH_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/310720251600"
EVAL_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/300720251000"

BEH_EPISODES_JSON_TRAIN = '/opt/ml/code/episodes/310720251600/310725_generated_rllib_ppo_rllib_seed_0000_2000eps_200steps_exp_0'
BEH_EPISODES_JSON_TEST = '/opt/ml/code/episodes/310720251600/310725_generated_rllib_ppo_rllib_seed_0000_200eps_200steps_exp_0'
BEH_EPISODES_JSON = '/opt/ml/code/episodes/310720251600/010825_generated_rllib_ppo_rllib_seed_0000_1000eps_200steps_exp_0'
EVAL_EPISODES_JSON = '/opt/ml/code/episodes/300720251000/050825_generated_rllib_ppo_rllib_seed_0000_1000eps_200steps_exp_0'
beh_agent = load_checkpoint(BEH_CHECKPOINT_PATH)
eval_agent = load_checkpoint(EVAL_CHECKPOINT_PATH)

# ---------------------------------------------------------------------------#
# 2.1. IS a mano
# --------------------------------------------------------------------------- #
reader_beh = JsonReader(BEH_EPISODES_JSON)
reader_beh_train = JsonReader(BEH_EPISODES_JSON_TRAIN)
reader_beh_test = JsonReader(BEH_EPISODES_JSON_TEST)
reader_target = JsonReader(EVAL_EPISODES_JSON)
beh_eps_df = load_json_to_df(reader_beh, 1000)
target_eps_df = load_json_to_df(reader_target, 2)
beh_expected_return = calculate_value_function(beh_eps_df, 0.99)
target_expected_return = calculate_value_function(target_eps_df, 0.99)
print(f"Avg_Expecting_Return (BEH_POLICY) Value - RLLIB Generated episodes: {beh_expected_return}")
print(f"Avg_Expecting_Return (TARGET_POLICY) Value - RLLIB Generated episodes: {target_expected_return}")

# Prepare logged dataset dictionary from the DataFrame
# (ensure obs/state and done flags are properly extracted as numpy arrays)
logged_dataset = {
    "state": np.stack(beh_eps_df["obs"].values),        # shape (total_steps, state_dim)
    "action": beh_eps_df["action"].astype(int).values,  # shape (total_steps,)
    "reward": beh_eps_df["reward"].values,              # shape (total_steps,)
    "done": beh_eps_df["done"].values,                  # boolean array indicating episode termination
    "pscore": beh_eps_df["action_prob"].values                # propensity scores (behavior policy probabilities)
}

# Initialize OPE with the desired estimators: Direct Method, Trajectory-wise IPS, and Doubly Robust
ope = OPE(
    logged_dataset=logged_dataset,
    ope_estimators=[DM(), TIS(), DR()]
)

# Compute target policy action probabilities for each state in the logged data
states = beh_eps_df["obs"]
input_dict = {"obs": states}
action_logits, _ = eval_agent.model(input_dict, [], None)
eval_action_probs = torch.softmax(action_logits, dim=1).numpy()


probs_ = torch.nn.Softmax(dim=1)
probs_ = probs_(logits)
prob = probs_[0][action]
target_prob = prob.detach().numpy()
behavior_prob = step["action_prob"]
states_tensor = torch.tensor(np.stack(df["obs"].values), dtype=torch.float32)
with torch.no_grad():
    # Get policy's action distribution for each state (shape: [num_steps, 4])
    action_logits = target_policy_model(states_tensor)           # forward pass (e.g., network outputs logits)
    eval_action_probs = torch.softmax(action_logits, dim=1).numpy()  # convert to probabilities

# Compute Q-hat for each state-action in the logged data using the fitted Q model
Q_model.eval()
with torch.no_grad():
    q_values = Q_model(states_tensor).numpy()  # shape: [num_steps, 4], Q_hat(s, a) for each action

# Build input for OPE: provide the target policy action distribution and estimated Q-values
input_dict = {
    "eval_policy": {  # you can name the evaluation policy as you like
        "action_dist": eval_action_probs,   # target policy's prob distribution over actions for each state
        "estimated_q_value": q_values       # model-estimated Q(s,a) for each state-action
    }
}

# Estimate policy values using the OPE estimators
ope_results = ope.estimate_policy_values(input_dict=input_dict)
print("OPE Estimates:", ope_results)
# Example output: {"direct_method":  ... , "trajectorywise_ips": ... , "doubly_robust": ...}
