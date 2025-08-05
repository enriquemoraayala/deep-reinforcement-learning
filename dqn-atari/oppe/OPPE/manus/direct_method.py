
import json
import numpy as np

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
        G += (gamma**t) * step["reward"]
    return G

def estimate_q_function(episodes_data, gamma=0.99):
    """
    Estimates the Q-function from the observed episodes using a simple averaging approach.
    This is a basic implementation for tabular Q-function estimation.
    For continuous state/action spaces, a regression model would be needed.

    Args:
        episodes_data (list): A list of episodes, where each episode is a list of steps.
        gamma (float): Discount factor.

    Returns:
        dict: A dictionary representing the estimated Q-function, mapping (state, action) tuples to their Q-values.
    """
    q_values = {}
    state_action_counts = {}

    for episode in episodes_data:
        # Calculate returns for each step in the episode
        returns_from_step = []
        for i in range(len(episode)):
            returns_from_step.append(calculate_return(episode[i:], gamma))

        for i, step in enumerate(episode):
            state = tuple(step["state"]) if isinstance(step["state"], list) else step["state"]
            action = step["action"]
            sa_pair = (state, action)

            if sa_pair not in q_values:
                q_values[sa_pair] = 0.0
                state_action_counts[sa_pair] = 0

            q_values[sa_pair] += returns_from_step[i]
            state_action_counts[sa_pair] += 1
    
    for sa_pair in q_values:
        if state_action_counts[sa_pair] > 0:
            q_values[sa_pair] /= state_action_counts[sa_pair]

    return q_values

def direct_method_ope(episodes_data, target_policy_probs, gamma=0.99):
    """
    Implements the Direct Method (DM) for Off-policy Policy Evaluation.

    Args:
        episodes_data (list): A list of episodes, where each episode is a list of steps.
        target_policy_probs (dict): A dictionary mapping (state, action) tuples to their probabilities under the target policy.
        gamma (float): Discount factor.

    Returns:
        float: The estimated value of the target policy.
    """
    # Step 1: Estimate the Q-function from the observed data
    estimated_q_function = estimate_q_function(episodes_data, gamma)

    # Step 2: Estimate the value of the target policy using the estimated Q-function
    # This assumes we can sample states from the observed data and apply the target policy
    # For simplicity, we'll average the expected Q-values over the initial states of episodes.
    
    total_estimated_value = 0.0
    num_episodes = len(episodes_data)

    if num_episodes == 0:
        return 0.0

    for episode in episodes_data:
        if not episode:
            continue
        
        initial_state = tuple(episode[0]["state"]) if isinstance(episode[0]["state"], list) else episode[0]["state"]
        
        # Calculate expected Q-value for the initial state under the target policy
        expected_q_for_initial_state = 0.0
        possible_actions = set([action for (s, action) in estimated_q_function if s == initial_state])
        
        for action in possible_actions:
            sa_pair = (initial_state, action)
            if sa_pair in target_policy_probs and sa_pair in estimated_q_function:
                target_prob = target_policy_probs[sa_pair]
                q_value = estimated_q_function[sa_pair]
                expected_q_for_initial_state += target_prob * q_value
            # If target_policy_probs doesn't have an entry for this (state, action), 
            # it means the target policy doesn't take this action from this state, so its probability is 0.
            # If estimated_q_function doesn't have an entry, it means we haven't observed this (state, action) pair.
            # In a real scenario, you might need to handle unseen state-action pairs (e.g., with a default value or generalization).

        total_estimated_value += expected_q_for_initial_state

    return total_estimated_value / num_episodes

if __name__ == '__main__':
    # Dummy episode data (same as importance_sampling.py for consistency)
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

    # Dummy target policy probabilities (state, action) -> probability
    dummy_target_policy_probs = {
        (0, 0): 0.7,
        (0, 1): 0.3,
        (1, 0): 0.05,
        (1, 1): 0.95
    }

    # Calculate DM
    dm_value = direct_method_ope(dummy_episodes_data, dummy_target_policy_probs)
    print(f"Direct Method (DM) Value: {dm_value}")

    # Example with a different episode to test Q-function estimation
    dummy_episodes_data_2 = [
        [
            {'state': 0, 'action': 0, 'reward': 10, 'behavior_policy_prob': 0.8, 'target_policy_prob': 0.7},
            {'state': 1, 'action': 1, 'reward': 1, 'behavior_policy_prob': 0.9, 'target_policy_prob': 0.95}
        ],
        [
            {'state': 0, 'action': 0, 'reward': 12, 'behavior_policy_prob': 0.2, 'target_policy_prob': 0.3},
            {'state': 1, 'action': 1, 'reward': 2, 'behavior_policy_prob': 0.1, 'target_policy_prob': 0.05}
        ]
    ]
    dm_value_2 = direct_method_ope(dummy_episodes_data_2, dummy_target_policy_probs)
    print(f"Direct Method (DM) Value (Episode 2): {dm_value_2}")

    # Note: For a real RLLib PPO policy, you would need to:
    # 1. Load the trained PPO policy.
    # 2. For each (state, action) pair in your episodes, query the PPO policy
    #    to get the action probabilities under the target policy.
    # 3. The `estimate_q_function` is a very basic tabular approach. For complex environments,
    #    you would replace this with a proper function approximator (e.g., a neural network).


