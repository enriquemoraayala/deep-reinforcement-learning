#!/usr/bin/env python3
"""
Train PPO on LunarLander-v3 with RLlib 2.11.

What it does
------------
- Uses a PPO config tuned as a strong starting point for LunarLander-v3.
- Trains with RLlib's Algorithm API (manual loop, no Tune dependency required).
- Runs deterministic evaluation every N training iterations.
- Stops early when evaluation reward stays above the solve threshold.
- Saves periodic checkpoints and the best checkpoint.
- Logs to RLlib's default logdir so you can inspect training in TensorBoard.

Suggested install
-----------------
pip install "ray[rllib]==2.11.0" "gymnasium[box2d]" tensorboard

Suggested run
-------------
python lunarlander_rllib_ppo.py

TensorBoard
-----------
tensorboard --logdir <tensorboard-logdir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import ray
from ray.rllib.algorithms.ppo import PPOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on LunarLander-v3 with RLlib 2.11."
    )

    # General experiment settings
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--framework", type=str, default="torch", choices=["torch", "tf2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--solve-reward", type=float, default=200.0)
    parser.add_argument("--solve-streak", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=80)
    parser.add_argument("--checkpoint-dir", type=str, default="/opt/ml/code/checkpoints/060420261500")
    parser.add_argument("--results-csv", type=str, default="/opt/ml/code/results/060420261500/lunarlander_ppo_results.csv")
    # parser.add_argument("--tensorboard-logdir", type=str, default=os.path.join(os.path.expanduser("~"), "ray_results"))
    parser.add_argument("--tensorboard-logdir", type=str, default="/opt/ml/code/traininglogs")

    # Environment difficulty / stochasticity
    parser.add_argument("--gravity", type=float, default=-10.0)
    parser.add_argument("--wind-power", type=float, default=15.0)
    parser.add_argument("--turbulence-power", type=float, default=1.5)
    parser.add_argument("--enable-wind", dest="enable_wind", action="store_true", default=True)
    parser.add_argument("--disable-wind", dest="enable_wind", action="store_false")

    # Resources / sampling
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--num-rollout-workers", type=int, default=4)
    parser.add_argument("--num-envs-per-worker", type=int, default=4)
    parser.add_argument("--rollout-fragment-length", type=int, default=1024)

    # Evaluation
    parser.add_argument("--evaluation-num-workers", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=40)

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--vf-loss-coeff", type=float, default=0.5)
    parser.add_argument("--vf-clip-param", type=float, default=100.0)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--sgd-minibatch-size", type=int, default=64)
    parser.add_argument("--num-sgd-iter", type=int, default=4)

    # Model
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--vf-share-layers", action="store_true", default=False)

    args = parser.parse_args()

    # Keep train_batch_size consistent with the rollout topology.
    args.train_batch_size = (
        args.num_rollout_workers
        * args.num_envs_per_worker
        * args.rollout_fragment_length
    )

    return args


def get_nested(d: dict[str, Any], path: list[str]) -> Optional[Any]:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def find_first_numeric(d: Any, candidate_paths: list[list[str]]) -> Optional[float]:
    for path in candidate_paths:
        value = get_nested(d, path) if isinstance(d, dict) else None
        if isinstance(value, (int, float)):
            return float(value)
    return None


def extract_train_reward(result: dict[str, Any]) -> Optional[float]:
    # RLlib has changed metric nesting a few times; try the most common variants.
    candidates = [
        ["episode_reward_mean"],
        ["env_runners", "episode_return_mean"],
        ["sampler_results", "episode_reward_mean"],
    ]
    return find_first_numeric(result, candidates)


def extract_eval_reward(result: dict[str, Any]) -> Optional[float]:
    candidates = [
        ["evaluation", "episode_reward_mean"],
        ["evaluation", "env_runners", "episode_return_mean"],
        ["episode_reward_mean"],
        ["env_runners", "episode_return_mean"],
    ]
    return find_first_numeric(result, candidates)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def normalize_checkpoint_path(save_result: Any) -> str:
    # RLlib versions differ: sometimes save() returns a string path,
    # sometimes an object containing the path.
    if isinstance(save_result, str):
        return save_result
    if hasattr(save_result, "checkpoint") and hasattr(save_result.checkpoint, "path"):
        return str(save_result.checkpoint.path)
    if hasattr(save_result, "path"):
        return str(save_result.path)
    return str(save_result)


def find_resume_state(checkpoint_dir: Path) -> Optional[dict]:
    """Return saved training state if a valid checkpoint exists, otherwise None."""
    state_file = checkpoint_dir / "training_state.json"
    if not state_file.exists():
        return None
    with state_file.open("r", encoding="utf-8") as f:
        state = json.load(f)
    checkpoint_path = state.get("checkpoint_path")
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None
    return state


def main() -> None:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_csv = Path(args.results_csv).resolve()
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    config = (
        PPOConfig()
        .environment(
            env=args.env,
            env_config={
                "continuous": False,
                "gravity": args.gravity,
                "enable_wind": args.enable_wind,
                "wind_power": args.wind_power,
                "turbulence_power": args.turbulence_power,
            },
        )
        .framework(args.framework)
        .resources(num_gpus=args.num_gpus)
        .rollouts(
            num_rollout_workers=args.num_rollout_workers,
            num_envs_per_worker=args.num_envs_per_worker,
            rollout_fragment_length=args.rollout_fragment_length,
            batch_mode="truncate_episodes",
        )
        .training(
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.gae_lambda,
            clip_param=args.clip_param,
            entropy_coeff=args.entropy_coeff,
            vf_loss_coeff=args.vf_loss_coeff,
            vf_clip_param=args.vf_clip_param,
            grad_clip=args.grad_clip,
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            model={
                "fcnet_hiddens": [args.hidden_size, args.hidden_size],
                "fcnet_activation": "tanh",
                "vf_share_layers": args.vf_share_layers,
            },
        )
        .evaluation(
            evaluation_interval=None,  # manual evaluation in the loop
            evaluation_num_workers=args.evaluation_num_workers,
            evaluation_duration=args.eval_episodes,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": False},
        )
        .debugging(
            log_level="WARN",
            logger_config={
                "type": "ray.tune.logger.TBXLogger",
                "logdir": str(Path(args.tensorboard_logdir).expanduser().resolve()),
            },
        )
    )

    algo = config.build()

    # Resume from checkpoint if one exists in the checkpoint directory.
    resume_state = find_resume_state(checkpoint_dir)
    if resume_state:
        algo.restore(resume_state["checkpoint_path"])

    logdir = getattr(algo, "logdir", None)
    print("=" * 80)
    print("RLlib PPO LunarLander-v3")
    print("=" * 80)
    print(f"Environment           : {args.env}")
    print(f"Framework             : {args.framework}")
    print(f"Enable wind           : {args.enable_wind}")
    print(f"Gravity               : {args.gravity}")
    print(f"Wind power            : {args.wind_power}")
    print(f"Turbulence power      : {args.turbulence_power}")
    print(f"Train batch size      : {args.train_batch_size}")
    print(f"Rollout workers       : {args.num_rollout_workers}")
    print(f"Envs per worker       : {args.num_envs_per_worker}")
    print(f"Rollout fragment len  : {args.rollout_fragment_length}")
    print(f"SGD minibatch size    : {args.sgd_minibatch_size}")
    print(f"Num SGD iter (epochs) : {args.num_sgd_iter}")
    print(f"Logdir                : {logdir if logdir else '(default RLlib logdir)'}")
    print(f"Checkpoint dir        : {checkpoint_dir}")
    print(f"Results CSV           : {results_csv}")
    print(f"TensorBoard           : tensorboard --logdir {args.tensorboard_logdir}")
    if resume_state:
        print(f"Resuming from         : {resume_state['checkpoint_path']} (iter {resume_state.get('iteration', '?')})")
    print("=" * 80)

    best_eval_reward: float = resume_state.get("best_eval_reward", float("-inf")) if resume_state else float("-inf")
    best_checkpoint_path: Optional[str] = resume_state.get("best_checkpoint_path") if resume_state else None
    solve_streak: int = resume_state.get("solve_streak", 0) if resume_state else 0
    start_iteration: int = resume_state.get("iteration", 0) + 1 if resume_state else 1
    start_time = time.time()

    fieldnames = [
        "iteration",
        "elapsed_sec",
        "train_reward_mean",
        "eval_reward_mean",
        "best_eval_reward",
        "checkpoint_path",
    ]

    csv_mode = "a" if resume_state else "w"
    with results_csv.open(csv_mode, newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        if not resume_state:
            writer.writeheader()

        for iteration in range(start_iteration, args.max_iters + 1):
            train_result = algo.train()
            train_reward = extract_train_reward(train_result)

            eval_reward: Optional[float] = None
            ckpt_path: Optional[str] = None

            # Manual evaluation every N iterations.
            if iteration % args.eval_every == 0:
                eval_result = algo.evaluate()
                eval_reward = extract_eval_reward(eval_result)

                if eval_reward is not None and eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    save_result = algo.save(checkpoint_dir=str(checkpoint_dir / "best"))
                    best_checkpoint_path = normalize_checkpoint_path(save_result)
                    save_json(
                        checkpoint_dir / "best" / "best_checkpoint_meta.json",
                        {
                            "iteration": iteration,
                            "best_eval_reward": best_eval_reward,
                            "checkpoint_path": best_checkpoint_path,
                        },
                    )
                    ckpt_path = best_checkpoint_path
                    save_json(
                        checkpoint_dir / "training_state.json",
                        {
                            "iteration": iteration,
                            "best_eval_reward": best_eval_reward,
                            "best_checkpoint_path": best_checkpoint_path,
                            "solve_streak": solve_streak,
                            "checkpoint_path": best_checkpoint_path,
                        },
                    )

                if eval_reward is not None and eval_reward >= args.solve_reward:
                    solve_streak += 1
                else:
                    solve_streak = 0

            # Periodic checkpoints regardless of evaluation.
            if iteration % args.checkpoint_every == 0:
                periodic = algo.save(checkpoint_dir=str(checkpoint_dir / f"iter_{iteration:04d}"))
                ckpt_path = normalize_checkpoint_path(periodic)
                save_json(
                    checkpoint_dir / "training_state.json",
                    {
                        "iteration": iteration,
                        "best_eval_reward": None if best_eval_reward == float("-inf") else best_eval_reward,
                        "best_checkpoint_path": best_checkpoint_path,
                        "solve_streak": solve_streak,
                        "checkpoint_path": ckpt_path,
                    },
                )

            elapsed = time.time() - start_time
            print(
                f"[iter {iteration:03d}] "
                f"train_mean={train_reward if train_reward is not None else 'n/a'} | "
                f"eval_mean={eval_reward if eval_reward is not None else 'n/a'} | "
                f"best_eval={best_eval_reward if best_eval_reward > -1e18 else 'n/a'} | "
                f"solve_streak={solve_streak}/{args.solve_streak}"
            )

            writer.writerow(
                {
                    "iteration": iteration,
                    "elapsed_sec": round(elapsed, 2),
                    "train_reward_mean": train_reward,
                    "eval_reward_mean": eval_reward,
                    "best_eval_reward": None if best_eval_reward == float("-inf") else best_eval_reward,
                    "checkpoint_path": ckpt_path,
                }
            )
            f_csv.flush()

            # Early stop when the solve threshold is stable.
            if solve_streak >= args.solve_streak:
                print("\nEnvironment solved consistently. Stopping early.")
                break

    final_checkpoint = algo.save(checkpoint_dir=str(checkpoint_dir / "final"))
    final_checkpoint_path = normalize_checkpoint_path(final_checkpoint)

    summary = {
        "env": args.env,
        "framework": args.framework,
        "env_config": {
            "continuous": False,
            "gravity": args.gravity,
            "enable_wind": args.enable_wind,
            "wind_power": args.wind_power,
            "turbulence_power": args.turbulence_power,
        },
        "train_batch_size": args.train_batch_size,
        "num_sgd_iter": args.num_sgd_iter,
        "solve_reward": args.solve_reward,
        "solve_streak": args.solve_streak,
        "best_eval_reward": None if best_eval_reward == float("-inf") else best_eval_reward,
        "best_checkpoint_path": best_checkpoint_path,
        "final_checkpoint_path": final_checkpoint_path,
        "logdir": logdir,
        "results_csv": str(results_csv),
    }
    save_json(checkpoint_dir / "run_summary.json", summary)

    print("\nDone.")
    if best_checkpoint_path:
        print(f"Best checkpoint : {best_checkpoint_path}")
    print(f"Final checkpoint: {final_checkpoint_path}")
    print(f"TensorBoard log : tensorboard --logdir {args.tensorboard_logdir}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
