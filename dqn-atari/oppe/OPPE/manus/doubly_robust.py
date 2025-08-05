
import json
import numpy as np

# Re-using functions from importance_sampling.py and direct_method.py
# In a real project, these would likely be in a shared utility file or imported.

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
        
        if (state, action) not in target_policy_probs or (state, action) not in behavior_policy_probs:
            # This can happen if the state-action pair was not encountered during target/behavior policy training
            # or if the policy dictionaries are incomplete. For robustness, we can assume 0 probability.
            # In a real scenario, you might want to log a warning or handle this more explicitly.
            target_prob = 0.0
            behavior_prob = 0.0
        else:
            target_prob = target_policy_probs[(state, action)]
            behavior_prob = behavior_policy_probs[(state, action)]

        if behavior_prob == 0:
            # If behavior policy never takes this action, importance ratio is undefined or infinite.
            # For practical purposes in OPE, this usually means the trajectory is not covered.
            # Returning 0.0 or raising an error depends on the desired behavior for uncovered trajectories.
            # Here, we return 0.0, effectively giving it no weight.
            return 0.0 
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
            state = tuple(step['state']) if isinstance(step['state'], list) else step['state']
            action = step['action']
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

def estimate_v_function_from_q(q_function, target_policy_probs):
    """
    Estimates the V-function from an estimated Q-function and target policy probabilities.

    Args:
        q_function (dict): Estimated Q-function, mapping (state, action) to Q-value.
        target_policy_probs (dict): A dictionary mapping (state, action) tuples to their probabilities under the target policy.

    Returns:
        dict: Estimated V-function, mapping state to V-value.
    """
    v_function = {}
    for (state, action), q_val in q_function.items():
        if state not in v_function:
            v_function[state] = 0.0
        
        target_prob = target_policy_probs.get((state, action), 0.0)
        v_function[state] += target_prob * q_val
    return v_function

def doubly_robust_ope(episodes_data, target_policy_probs, behavior_policy_probs, gamma=0.99):
    """
    Implements the Doubly Robust (DR) method for Off-policy Policy Evaluation.

    Args:
        episodes_data (list): A list of episodes, where each episode is a list of steps.
        target_policy_probs (dict): A dictionary mapping (state, action) tuples to their probabilities under the target policy.
        behavior_policy_probs (dict): A dictionary mapping (state, action) tuples to their probabilities under the behavior policy.
        gamma (float): Discount factor.

    Returns:
        float: The estimated value of the target policy.
    """
    num_episodes = len(episodes_data)
    if num_episodes == 0:
        return 0.0

    # Step 1: Estimate Q-function from observed data (Direct Method component)
    estimated_q_function = estimate_q_function(episodes_data, gamma)
    estimated_v_function = estimate_v_function_from_q(estimated_q_function, target_policy_probs)

    dr_estimate = 0.0

    for episode in episodes_data:
        if not episode:
            continue

        # Initial state and action of the episode
        initial_state = tuple(episode[0]['state']) if isinstance(episode[0]['state'], list) else episode[0]['state']
        initial_action = episode[0]['action']
        initial_sa_pair = (initial_state, initial_action)

        # DM component for the initial state
        dm_initial_value = estimated_v_function.get(initial_state, 0.0)

        # IS component for the episode
        importance_ratio_episode = calculate_importance_ratio(episode, target_policy_probs, behavior_policy_probs)
        episode_return = calculate_return(episode, gamma)

        # Calculate the Q-value of the initial state-action pair under the target policy from the estimated Q-function
        # This is the Q(s_0, a_0) term in the DR formula
        q_s0_a0_model = estimated_q_function.get(initial_sa_pair, 0.0)

        # DR formula: V_DM + rho * (G - Q(s_0, a_0))
        # This is a simplified version for episodic tasks, focusing on the initial state.
        # A more general per-decision DR would iterate through all steps.
        dr_estimate += dm_initial_value + importance_ratio_episode * (episode_return - q_s0_a0_model)

    return dr_estimate / num_episodes

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

    # Dummy policy probabilities (state, action) -> probability
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

    # Calculate DR
    dr_value = doubly_robust_ope(dummy_episodes_data, dummy_target_policy_probs, dummy_behavior_policy_probs)
    print(f"Doubly Robust (DR) Value: {dr_value}")

    # Note: For a real RLLib PPO policy, you would need to:
    # 1. Load the trained PPO policy.
    # 2. For each (state, action) pair in your episodes, query the PPO policy
    #    to get the action probabilities under the target policy.
    # 3. Ensure your behavior policy probabilities are also available (e.g., from logging).
    # 4. The `estimate_q_function` is a very basic tabular approach. For complex environments,
    #    you would replace this with a proper function approximator (e.g., a neural network).
    #    The `estimate_v_function_from_q` also assumes a tabular setting.


