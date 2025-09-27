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

from oppe_utils import load_checkpoint, load_json_to_df, calculate_value_function, calculate_return
from oppe_utils import QNetwork
from statistics import mean, stdev
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.offline.json_reader import JsonReader

from ray.rllib.algorithms import Algorithm
from ray.rllib.policy.sample_batch import SampleBatch
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
generate_eps = 0

if debug == 1:
    # Escucha en el puerto 5678 (puedes cambiarlo)
    debugpy.listen(("0.0.0.0", 5678))
    print("Esperando debugger de VS Code para conectar...")
    debugpy.wait_for_client()


def calculate_importance_ratio(episode, target_policy, source= 'online'):
    """
    Calculates the importance ratio for a given episode.

    Args:
        episode (list): A list of dictionaries, where each dictionary represents a step in the episode.
                        Each step should contain 'state', 'action', 'reward', 'behavior_policy_prob', and 'target_policy_prob'.
        target_policy_probs (dict): A dictionary mapping (state, action) tuples to their probabilities under the target policy.
        behavior_policy_probs (dict): A dictionary mapping (state, action) tuples to their probabilities under the behavior policy.

    Returns:
        float: The importance ratio for the episode.
    """
    importance_ratio = 1.0
    # target_policy = target_policy.get_policy()
    for idx, step in episode.iterrows():
        state = step['obs']
        action = step['action']
        if source == 'online':
            state = torch.tensor(state).unsqueeze(0)
        else:
            state = torch.tensor(state)
        input_dict = {"obs": state}
        logits, _ = target_policy.model(input_dict, [], None)
        probs_ = torch.nn.Softmax(dim=1)
        probs_ = probs_(logits)
        prob = probs_[0][action]
        target_prob = prob.detach().numpy()
        behavior_prob = step["action_prob"]

        if behavior_prob == 0:
            # This case should ideally be handled by ensuring coverage, but for robustness:
            return 0.0 # Or raise an error, depending on desired behavior
        importance_ratio *= (target_prob / behavior_prob)
    return importance_ratio


def ordinary_importance_sampling(episodes_data, target_policy, gamma=0.99, source='online'):
    """
    Implements the Ordinary Importance Sampling (OIS) method for Off-policy Policy Evaluation.

    Args:
        episodes_data (list): A list of episodes, where each episode is a list of steps.
        target_policy_probs (dict): A dictionary mapping (state, action) tuples to their probabilities under the target policy.
        behavior_policy_probs (dict): A dictionary mapping (state, action) tuples to their probabilities under the behavior policy.
        gamma (float): Discount factor.

    Returns:
        float: The estimated value of the target policy.
    """
    total_weighted_return = 0.0
    num_episodes = episodes_data['ep'].nunique()
    print(f"Calculating IS for {num_episodes}")

    if num_episodes == 0:
        return 0.0

    episode_returns = []
    for ep in range(num_episodes):
        episode = episodes_data[episodes_data['ep']==ep]
        #colocar problemas de soporte

        importance_ratio = calculate_importance_ratio(episode, target_policy, source)
        episode_return = calculate_return(episode, gamma)
        episode_returns.append(episode_return)

        total_weighted_return += importance_ratio * episode_return
    
    avg_expected_return = mean(episode_returns)
    return total_weighted_return / num_episodes, avg_expected_return


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


def define_rllib_estimators(algo):
    dr_estimator = DoublyRobust(
        # policy=algo.get_policy(),
        policy=algo,
        gamma=0.99,
        q_model_config={"type": FQETorchModel, "n_iters": 10, "lr": 0.0005},
        )

    is_estimator = ImportanceSampling(
        policy=algo,
        gamma=0.99,
        epsilon_greedy=0.05
    )

    wis_estimator = WeightedImportanceSampling(
        policy=algo,
        gamma=0.99,
        epsilon_greedy=0.05
    )

    dm_estimator = DirectMethod(
        policy=algo,
        gamma=0.99,
        q_model_config={"type": FQETorchModel, "n_iters": 10, "lr": 0.0005},
    )

    return dr_estimator, is_estimator, dm_estimator


def train_rllib_dm_dr_models(BEH_EPISODES_JSON_TRAIN, reader_beh_train, dr_estimator, dm_estimator):
    print("\n⏳ Entrenando Q‑model DM and DR ...")
    for i in range(2000):
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
    
    BEH_EPISODES_JSON_TRAIN = '/opt/ml/code/episodes/120820251600/140825_generated_rllib_ppo_rllib_seed_0000_1000eps_200steps_exp_0'
    BEH_EPISODES_JSON_TEST = '/opt/ml/code/episodes/310720251600/310725_generated_rllib_ppo_rllib_seed_0000_2000eps_200steps_exp_7'
    BEH_EPISODES_JSON = '/opt/ml/code/episodes/120820251600/140825_generated_rllib_ppo_rllib_seed_0000_1000eps_200steps_exp_0'
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
    beh_eps_df = load_json_to_df(reader_beh, 1000)
    beh_test_df = load_json_to_df(reader_beh_test, 2000)
    target_eps_df = load_json_to_df(reader_target, 1000)
    beh_expected_return, beh_return_stdev = calculate_value_function(beh_eps_df, 0.99)
    target_expected_return, target_return_stdev = calculate_value_function(target_eps_df, 0.99)
    print(f"Avg_Expecting_Return (BEH_POLICY) Value - RLLIB Generated episodes: {beh_expected_return: .3f} - STD {beh_return_stdev: .3f}")
    print(f"Avg_Expecting_Return (TARGET_POLICY) Value - RLLIB Generated episodes: {target_expected_return: .3f} - STD {target_return_stdev: .3f}")



    # --------------------------------------------------------------------------- #
    # IS FROM SCRATCH
    # --------------------------------------------------------------------------- #
    is_oppe, avg_expected_return = ordinary_importance_sampling(beh_test_df, eval_agent, gamma=0.99, source='rllib')
    print("\n=======  RESULTADOS OPPE with Custom Library   =======")
    print(f"Ordinary Importance Sampling (IS Custom) Value: {is_oppe}")

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

    # --------------------------------------------------------------------------- #
    # RLLIB OPPE: 
    # 
    # ENTRENAR ESTIMADOR DM (requiere Q‑model)
    # --------------------------------------------------------------------------- #
    # reader_train = JsonReader(reader_beh)  # 200 transiciones/batch

    dr_estimator, is_estimator, dm_estimator = define_rllib_estimators(eval_agent)
    reader_beh_test = JsonReader(BEH_EPISODES_JSON_TEST)
    is_values = []
    for idx in range(2000):
        batch_test  = reader_beh_test.next()
        is_value = is_estimator.estimate(batch_test)
        is_values.append(is_value['v_target'])
    print(f"Ordinary Importance Sampling (IS RLLIB) Value: {mean(is_values)}")

    dm_estimator, dr_estimator = train_rllib_dm_dr_models(BEH_EPISODES_JSON_TRAIN, reader_beh_train, dr_estimator, dm_estimator)
    reader_beh_test = JsonReader(BEH_EPISODES_JSON_TEST)
    dr_ests = []
    dm_ests = []
    for i in range(2000):
            batch = reader_beh_test.next()
            dr_ests.append(dr_estimator.estimate(batch)['v_target'])
            dm_ests.append(dm_estimator.estimate(batch)['v_target'])

    print(f'Direct Method (DM) with RLLIB V_ expected estimation {mean(dm_ests)} and STD {stdev(dm_ests)}')
    print(f'Double Roboust (DR) with RLLIB V_ expected estimation {mean(dr_ests)} and STD {stdev(dr_ests)}')

    



if __name__ == '__main__':
    oppe()
