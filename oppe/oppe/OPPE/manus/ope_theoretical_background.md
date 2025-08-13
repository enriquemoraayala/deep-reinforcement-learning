# Off-policy Policy Evaluation (OPE) Methods: Theoretical Background

Off-policy Policy Evaluation (OPE) is a crucial task in reinforcement learning (RL) that aims to estimate the performance of a target policy using data collected from a different behavior (or logging) policy. This is particularly useful in real-world applications where it might be costly or unsafe to directly deploy and evaluate new policies. OPE allows for the offline assessment of policies, enabling informed decision-making before online deployment.

This document provides a theoretical overview of three prominent OPE methods: Importance Sampling (IS), Direct Method (DM), and Doubly Robust (DR). We will delve into their underlying principles, mathematical formulations, and discuss their respective strengths and weaknesses.

## 1. Importance Sampling (IS)

Importance Sampling is a widely used technique for OPE due to its simplicity and unbiasedness (under certain conditions). It re-weights the returns observed under the behavior policy to account for the difference in probabilities of trajectories under the target policy.

### 1.1. Core Concept

The fundamental idea behind Importance Sampling is to correct for the discrepancy between the data-generating distribution (behavior policy) and the distribution under which we want to evaluate the policy (target policy). This correction is achieved by assigning a weight to each observed trajectory or state-action pair. This weight, known as the importance ratio, is the ratio of the probability of observing the trajectory under the target policy to the probability of observing it under the behavior policy.

### 1.2. Mathematical Formulation

Let $\pi_b$ be the behavior policy and $\pi_t$ be the target policy. We want to estimate the expected return of the target policy, $V^{\pi_t}$. The data consists of trajectories collected under $\pi_b$: $\tau = (S_0, A_0, R_1, S_1, A_1, R_2, ..., S_{T-1}, A_{T-1}, R_T, S_T)$.

The importance ratio for a single step $(S_t, A_t)$ is given by:

$$ \rho_t = \frac{\pi_t(A_t|S_t)}{\pi_b(A_t|S_t)} $$

For a full trajectory, the importance ratio is the product of the individual step importance ratios:

$$ \rho_\tau = \prod_{t=0}^{T-1} \frac{\pi_t(A_t|S_t)}{\pi_b(A_t|S_t)} $$

The **Ordinary Importance Sampling (OIS)** estimator for the expected return of the target policy is:

$$ V_{OIS}(\pi_t) = \frac{1}{N} \sum_{i=1}^{N} \rho_{\tau_i} G_i $$

where $N$ is the number of trajectories, $\tau_i$ is the $i$-th trajectory, and $G_i$ is the return (sum of discounted rewards) of the $i$-th trajectory.

While OIS is unbiased, its variance can be very high, especially when the behavior and target policies differ significantly. To mitigate this, **Weighted Importance Sampling (WIS)** is often used. WIS uses a weighted average of the returns, where the weights are the importance ratios:

$$ V_{WIS}(\pi_t) = \frac{\sum_{i=1}^{N} \rho_{\tau_i} G_i}{\sum_{i=1}^{N} \rho_{\tau_i}} $$

WIS is generally biased but has lower variance than OIS. The bias of WIS diminishes as the number of trajectories increases.

### 1.3. Advantages and Disadvantages

**Advantages:**
*   **Unbiased (OIS):** If the behavior policy covers the target policy (i.e., $\pi_b(A|S) > 0$ whenever $\pi_t(A|S) > 0$), OIS provides an unbiased estimate of the target policy's value.
*   **Model-free:** IS methods do not require a model of the environment (e.g., transition probabilities or reward function).
*   **Conceptually simple:** The idea of re-weighting based on policy differences is intuitive.

**Disadvantages:**
*   **High Variance:** The primary drawback of IS methods, especially OIS, is their potentially high variance. This occurs when the importance ratios are large, which happens when the target policy takes actions that are very unlikely under the behavior policy.
*   **Requires Policy Probabilities:** Both $\pi_t(A|S)$ and $\pi_b(A|S)$ must be known for all observed state-action pairs.
*   **Off-policy Coverage:** If the target policy can take actions that the behavior policy never takes (i.e., $\pi_b(A|S) = 0$ while $\pi_t(A|S) > 0$), the importance ratio becomes undefined, and IS cannot be applied directly.

## 2. Direct Method (DM)

The Direct Method, also known as Model-Based OPE or Regression-Based OPE, approaches the problem by first learning a model of the environment's reward function or value function from the observed data. Once this model is learned, it is then used to estimate the value of the target policy.

### 2.1. Core Concept

Instead of re-weighting observed returns, the Direct Method attempts to directly estimate the value function or Q-function of the target policy. This is typically done by training a supervised learning model (e.g., a regression model) to predict rewards or values based on the observed states and actions. The learned model then allows for the direct computation of the target policy's value, often by simulating trajectories under the target policy using the learned model.

### 2.2. Mathematical Formulation

Let's consider estimating the Q-function, $Q^{\pi_t}(S, A)$, for the target policy. We can train a regression model, $\hat{Q}(S, A)$, using the observed transitions $(S_t, A_t, R_{t+1}, S_{t+1})$ and the Bellman equation as a target. For example, for a given state-action pair $(S, A)$, the target for the regression could be $R + \gamma \max_{A'} \hat{Q}(S', A')$.

Once $\hat{Q}$ is learned, the value of the target policy can be estimated by:

$$ V_{DM}(\pi_t) = \mathbb{E}_{S \sim d^{\pi_t}} [\sum_A \pi_t(A|S) \hat{Q}(S, A)] $$

where $d^{\pi_t}$ is the stationary state distribution under the target policy. In practice, this expectation is often approximated by averaging over observed states from the behavior policy and then applying the target policy.

Alternatively, one can directly model the reward function, $\hat{R}(S, A)$, and potentially the transition dynamics, $\hat{P}(S'|S, A)$. Then, the value of the target policy can be estimated by simulating trajectories using these learned models.

### 2.3. Advantages and Disadvantages

**Advantages:**
*   **Low Variance:** DM methods tend to have lower variance compared to IS methods because they smooth out the noise in the observed returns by learning a model.
*   **Does not require Policy Probabilities:** Unlike IS, DM does not explicitly require the probabilities of actions under the behavior or target policies during the evaluation phase (though they might be implicitly used during model training if the model is policy-dependent).
*   **Handles Off-policy Coverage Issues:** DM can potentially handle situations where the target policy takes actions not seen under the behavior policy, as long as the learned model can generalize to those unseen state-action pairs.

**Disadvantages:**
*   **Bias:** The main disadvantage of DM is its potential for bias. The accuracy of the OPE estimate heavily relies on the accuracy of the learned model. If the model is misspecified or inaccurate, the estimate will be biased.
*   **Model Dependence:** Requires learning an accurate model of the environment (reward function, value function, or transition dynamics), which can be challenging, especially in complex environments or with limited data.
*   **Computational Cost:** Training a model can be computationally expensive.

## 3. Doubly Robust (DR)

The Doubly Robust estimator combines the strengths of both Importance Sampling and the Direct Method, aiming to achieve both low bias and low variance. It is 


called "doubly robust" because it provides an unbiased estimate of the target policy's value if either the importance ratios are correct (like in IS) or the learned model is correct (like in DM).

### 3.1. Core Concept

The DR estimator constructs an estimate by taking the direct method's estimate and correcting it with a term that involves the importance-sampled residuals of the learned model. The idea is that if the model is accurate, the residuals will be small, and the estimate will be close to the DM estimate. If the model is inaccurate, the importance-sampling term will correct for the model's errors, leading to an unbiased estimate.

### 3.2. Mathematical Formulation

The Doubly Robust estimator for the value of the target policy is given by:

$$ V_{DR}(\pi_t) = V_{DM}(\pi_t) + \frac{1}{N} \sum_{i=1}^{N} \rho_{\tau_i} (G_i - Q^{\pi_t}(S_0, A_0)) $$

where $V_{DM}(\pi_t)$ is the direct method estimate, $\rho_{\tau_i}$ is the importance ratio for trajectory $i$, $G_i$ is the observed return for trajectory $i$, and $Q^{\pi_t}(S_0, A_0)$ is the estimated Q-value of the initial state-action pair of trajectory $i$ under the target policy, obtained from the learned model.

An alternative formulation, often used in practice, is the per-decision DR estimator:

$$ V_{DR}(\pi_t) = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T-1} \gamma^t \left( \rho_t \left( R_{t+1} + \gamma V^{\pi_t}(S_{t+1}) - Q^{\pi_t}(S_t, A_t) \right) + V^{\pi_t}(S_t) \right) $$

where $\rho_t$ is the importance ratio at time step $t$, and $V^{\pi_t}$ and $Q^{\pi_t}$ are the value and Q-value functions estimated by the direct method.

### 3.3. Advantages and Disadvantages

**Advantages:**
*   **Doubly Robust Property:** The key advantage is that the DR estimator is unbiased if either the importance ratios are correct or the learned model is correct. This makes it more robust to errors in either component.
*   **Low Variance:** By incorporating a model, the DR estimator can achieve lower variance than IS methods.
*   **Low Bias:** By incorporating importance sampling, the DR estimator can have lower bias than DM methods, especially when the model is misspecified.

**Disadvantages:**
*   **Complexity:** The DR estimator is more complex to implement than either IS or DM alone.
*   **Requires Both Policy Probabilities and a Model:** It requires both the policy probabilities for the importance ratios and a learned model for the direct method component.
*   **Potential for High Variance:** Although generally lower than IS, the variance of the DR estimator can still be high if both the model is poor and the importance ratios are large.

## 4. Summary and Conclusion

Off-policy Policy Evaluation is a critical component of developing and deploying safe and effective reinforcement learning systems. Importance Sampling, the Direct Method, and Doubly Robust estimators represent three key approaches to this problem, each with its own set of trade-offs.

*   **Importance Sampling** is simple and unbiased but can suffer from high variance.
*   **The Direct Method** has low variance but can be biased if the learned model is inaccurate.
*   **The Doubly Robust** method combines the best of both worlds, offering a more robust solution that can be both low-bias and low-variance.

The choice of which OPE method to use depends on the specific application, the quality of the available data, and the computational resources available. For a more in-depth understanding and practical applications, readers are encouraged to explore the references below.

## 5. References

1.  Farajtabar, M., Chow, Y., & Ghavamzadeh, M. (2018). More Robust Doubly Robust Off-policy Evaluation. *Proceedings of the 35th International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 80:1447-1456 Available from https://proceedings.mlr.press/v80/farajtabar18a.html
2.  Jiang, N., & Li, L. (2016). Doubly Robust Off-Policy Value Evaluation for Reinforcement Learning. *Proceedings of the 33rd International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 48:652-661 Available from https://proceedings.mlr.press/v48/jiang16.html
3.  Thomas, P., & Brunskill, E. (2016). Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning. *Proceedings of the 33rd International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 48:2139-2148 Available from https://proceedings.mlr.press/v48/thomas16.html


