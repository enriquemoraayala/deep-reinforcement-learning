import pandas as pd
from ray.rllib.offline.json_reader import JsonReader
from oppe_utils import load_json_to_df

BEH_EPISODES_JSON = "/opt/ml/code/episodes/120820251600/140825_generated_rllib_ppo_rllib_seed_0000_1000eps_200steps_exp_0/output-2025-08-14_12-19-12_worker-0_0.json"  
reader_beh = JsonReader(BEH_EPISODES_JSON)
df = load_json_to_df(reader_beh, 1000)

episode_lengths = df.groupby("ep")["step"].count()

# Calcular la recompensa total por episodio
episode_rewards = df.groupby("ep")["reward"].sum()

# Calcular estadísticas básicas
stats = {
    "num_episodes": df["ep"].nunique(),
    "length_min": episode_lengths.min(),
    "length_max": episode_lengths.max(),
    "length_avg": episode_lengths.mean(),
    "reward_min": episode_rewards.min(),
    "reward_max": episode_rewards.max(),
    "reward_avg": episode_rewards.mean(),
    "reward_std": episode_rewards.std(),
    "episodes_lt_200_steps": (episode_lengths < 200).sum(),
    "episodes_ge_200_steps": (episode_lengths >= 200).sum(),
}

# Mostrar resultados
print("📊 Estadísticas de los episodios:")
for k, v in stats.items():
    print(f"{k:>20}: {v:.3f}" if isinstance(v, (float, int)) else f"{k:>20}: {v}")

# Si quieres un resumen tabular:
summary_df = pd.DataFrame({
    "episode_length": episode_lengths,
    "total_reward": episode_rewards
})
print("\nResumen por episodio:")
print(summary_df.describe())
