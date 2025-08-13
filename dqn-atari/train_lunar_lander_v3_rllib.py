# Requisitos:
#   pip install "ray[rllib]==2.11.0" gymnasium "gymnasium[box2d]"
#   (Asegúrate de tener Box2D para LunarLander)

import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig

def main():
    # Inicializa Ray en local
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    # --- Configuración PPO para un solo worker ---
    config = (
        PPOConfig()
        .environment("LunarLander-v3")            # Gymnasium 1.0
        .framework("torch")
        .rollouts(
            num_rollout_workers=0,                # ← un solo worker (el local)
            num_envs_per_worker=8,                # vectorización en el worker local
            rollout_fragment_length=1024          # fragmentos largos para reducir overhead
        )
        .training(
            train_batch_size=16384,               # tamaño del batch total (por iteración)
            sgd_minibatch_size=2048,
            num_sgd_iter=10,                      # ← ÉPOCAS de PPO por iteración
            gamma=0.99,
            lambda_=0.95,
            lr=5e-4,
            vf_clip_param=10.0,
            clip_param=0.2,
            entropy_coeff=0.01,
            grad_clip=0.5,
            model={"fcnet_hiddens": [256, 256], "fcnet_activation": "tanh"},
            lr_schedule=[[0, 5e-4], [6_000_000, 0.0]]
        )
        .resources(num_gpus=0)                    # CPU-only por defecto
    )

    algo = config.build()

    # --- Entrenamiento + guardado de checkpoints ---
    checkpoint_dir = "/opt/ml/code/checkpoints/130820251600"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_reward = float("-inf")
    recent_means = []
    patience = 5
    target_timesteps = 6_000_000

    print("Entrenando PPO (un solo worker). Épocas por iteración (num_sgd_iter): 10")
    for i in range(10000):  # bucle suficientemente grande; salimos por criterios abajo
        result = algo.train()

        # RLlib puede exponer distintas claves según versión; probamos ambas
        total_ts = result.get("num_env_steps_trained", result.get("timesteps_total", 0))
        mean_r = result.get("episode_reward_mean", float("nan"))

        # Guardar mejor checkpoint cuando mejora la media
        if mean_r > best_reward:
            best_reward = mean_r
            ckpt_path = algo.save(checkpoint_dir)
            print(f"[iter {i:04d}] Nueva mejor media: {best_reward:.2f} | "
                  f"ts={int(total_ts)} | checkpoint: {ckpt_path}")

        # Log básico
        print(f"iter {i:04d} | reward_mean={mean_r:.2f} | timesteps={int(total_ts)}")

        # Ventana reciente para criterio de "resuelto"
        recent_means.append(mean_r)
        if len(recent_means) > patience:
            recent_means.pop(0)

        solved = len(recent_means) == patience and all(r >= 200 for r in recent_means)
        if solved and total_ts >= 300_000:
            print("Criterio de parada: entorno resuelto (≥200 en las últimas 5 iteraciones).")
            break

        if total_ts >= target_timesteps:
            print("Criterio de parada: alcanzado el objetivo de ~6M timesteps.")
            break

    # Guardar un checkpoint final
    final_ckpt = algo.save(checkpoint_dir)
    print(f"Checkpoint final guardado en: {final_ckpt}")

    # (Opcional) imprimir ruta del mejor checkpoint si quieres persistirla en un archivo
    with open(os.path.join(checkpoint_dir, "LAST_CHECKPOINT.txt"), "w") as f:
        f.write(str(final_ckpt))

    ray.shutdown()

if __name__ == "__main__":
    main()