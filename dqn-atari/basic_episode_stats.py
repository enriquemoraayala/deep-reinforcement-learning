import pandas as pd
import debugpy
from ray.rllib.offline.json_reader import JsonReader
from oppe_utils import load_json_to_df, load_json_to_df_max

debug = 1

if debug == 1:
    # Escucha en el puerto 5678 (puedes cambiarlo)
    debugpy.listen(("0.0.0.0", 5678))
    print("Esperando debugger de VS Code para conectar...")
    debugpy.wait_for_client()

BEH_EPISODES_JSON = "/opt/ml/code/episodes/120820251600/011125_01_generated_rllib_ppo_rllib_seed_0000_10000eps_300steps_exp_0"
reader_beh = JsonReader(BEH_EPISODES_JSON)
df, eps, steps = load_json_to_df_max(reader_beh, 100000)

print(f"Transformed {eps} episodes with a total of {steps}")

episode_lengths = df.groupby("ep")["step"].count()

# Calcular la recompensa total por episodio
episode_rewards = df.groupby("ep")["reward"].sum()

# Contar cuántos True y False hay en toda la tabla
terminated_counts = df["terminated"].value_counts(dropna=False)
truncated_counts = df["truncated"].value_counts(dropna=False)

combo_counts = pd.crosstab(df["terminated"], df["truncated"])
print(combo_counts)


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
    "episodes_lt_300_steps": (episode_lengths < 300).sum(),
    "episodes_g_300_steps": (episode_lengths > 300).sum(),
    "episodes_e_300_steps": (episode_lengths == 300).sum(),
    "truncated": truncated_counts,
    "terminated": terminated_counts,
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
