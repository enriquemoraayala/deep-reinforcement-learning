
import json
import numpy as np

def calculate_importance_ratio(episode, target_policy_probs, behavior_policy_probs):
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
    for step in episode:
        state = tuple(step['state']) if isinstance(step['state'], list) else step['state']
        action = step['action']
        
        # Ensure the keys exist in the probability dictionaries
        if (state, action) not in target_policy_probs or (state, action) not in behavior_policy_probs:
            raise ValueError(f"Policy probabilities for state {state} and action {action} not found.")

        target_prob = target_policy_probs[(state, action)]
        behavior_prob = behavior_policy_probs[(state, action)]

        if behavior_prob == 0:
            # This case should ideally be handled by ensuring coverage, but for robustness:
            return 0.0 # Or raise an error, depending on desired behavior
        importance_ratio *= (target_prob / behavior_prob)
    return importance_ratio

def calculate_return(episode, gamma=0.99):
    """
    Calculates the discounted return for a given episode.

    Args:
        episode (list): A list of dictionaries, where each dictionary represents a step in the episode.
        gamma (float): Discount factor.

    Returns:
        float: The discounted return for the episode.
    """
    G = 0
    for t, step in enumerate(episode):
        G += (gamma**t) * step['reward']
    return G

def ordinary_importance_sampling(episodes_data, target_policy_probs, behavior_policy_probs, gamma=0.99):
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
    num_episodes = len(episodes_data)

    if num_episodes == 0:
        return 0.0

    for episode in episodes_data:
        importance_ratio = calculate_importance_ratio(episode, target_policy_probs, behavior_policy_probs)
        episode_return = calculate_return(episode, gamma)
        total_weighted_return += importance_ratio * episode_return

    return total_weighted_return / num_episodes

def weighted_importance_sampling(episodes_data, target_policy_probs, behavior_policy_probs, gamma=0.99):
    """
    Implements the Weighted Importance Sampling (WIS) method for Off-policy Policy Evaluation.

    Args:
        episodes_data (list): A list of episodes, where each episode is a list of steps.
        target_policy_probs (dict): A dictionary mapping (state, action) tuples to their probabilities under the target policy.
        behavior_policy_probs (dict): A dictionary mapping (state, action) tuples to their probabilities under the behavior policy.
        gamma (float): Discount factor.

    Returns:
        float: The estimated value of the target policy.
    """
    sum_of_weighted_returns = 0.0
    sum_of_importance_ratios = 0.0

    for episode in episodes_data:
        importance_ratio = calculate_importance_ratio(episode, target_policy_probs, behavior_policy_probs)
        episode_return = calculate_return(episode, gamma)

        sum_of_weighted_returns += importance_ratio * episode_return
        sum_of_importance_ratios += importance_ratio

    if sum_of_importance_ratios == 0:
        return 0.0  # Avoid division by zero

    return sum_of_weighted_returns / sum_of_importance_ratios

if __name__ == '__main__':
    # Example Usage (replace with actual JSON loading and policy data)
    # This is a dummy example to show the expected input format.
    # In a real scenario, episodes_data would come from your JSON input,
    # and target/behavior_policy_probs would be derived from your RLLib PPO policy.

    # Dummy episode data
    dummy_episodes_data = [
        [
            {'state': 0, 'action': 0, 'reward': 1, 'behavior_policy_prob': 0.8, 'target_policy_prob': 0.7},
            {'state': 1, 'action': 1, 'reward': 10, 'behavior_policy_prob': 0.9, 'target_policy_prob': 0.95}
        ],
        [
            {'state': 0, 'action': 1, 'reward': 2, 'behavior_policy_prob': 0.2, 'target_policy_prob': 0.3},
            {'state': 1, 'action': 0, 'reward': 5, 'behavior_policy_prob': 0.1, 'target_policy_prob': 0.05}
        ]
    ]

    # Dummy policy probabilities (state, action) -> probability
    # In a real scenario, these would be derived from your RLLib PPO policy
    dummy_target_policy_probs = {
        (0, 0): 0.7,
        (0, 1): 0.3,
        (1, 0): 0.05,
        (1, 1): 0.95
    }

    dummy_behavior_policy_probs = {
        (0, 0): 0.8,
        (0, 1): 0.2,
        (1, 0): 0.1,
        (1, 1): 0.9
    }

    # Calculate OIS
    ois_value = ordinary_importance_sampling(dummy_episodes_data, dummy_target_policy_probs, dummy_behavior_policy_probs)
    print(f"Ordinary Importance Sampling (OIS) Value: {ois_value}")

    # Calculate WIS
    wis_value = weighted_importance_sampling(dummy_episodes_data, dummy_target_policy_probs, dummy_behavior_policy_probs)
    print(f"Weighted Importance Sampling (WIS) Value: {wis_value}")

    # Example of loading from JSON (conceptual)
    # with open('episodes.json', 'r') as f:
    #     episodes_from_json = json.load(f)
    # ois_value_json = ordinary_importance_sampling(episodes_from_json, target_policy_probs, behavior_policy_probs)
    # print(f"OIS Value from JSON: {ois_value_json}")

    # Note: For a real RLLib PPO policy, you would need to:
    # 1. Load the trained PPO policy.
    # 2. For each (state, action) pair in your episodes, query the PPO policy
    #    to get the action probabilities under the target policy.
    # 3. Ensure your behavior policy probabilities are also available (e.g., from logging).



