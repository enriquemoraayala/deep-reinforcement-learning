"""
Azure ML-ready training script for FQTE / FQE (GPU-ready).

What this script guarantees (GPU readiness):
- Uses CUDA automatically when available (or force with --device cuda).
- Moves Q-networks to GPU and moves each training batch to GPU.
- Supports mixed precision AMP on CUDA (--amp) for speed.
- Resume training loads latest checkpoint and moves optimizer state to the right device.
- RLlib policy inference is kept on CPU numpy (typical RLlib expectation), while Q-model trains on GPU.

Typical AML command example:
  python train_fqte_aml_gpu.py \
    --beh_checkpoint_dir ${{inputs.beh_ckpt}} \
    --eval_checkpoint_dir ${{inputs.eval_ckpt}} \
    --episodes_root ${{inputs.episodes}} \
    --beh_train_rel 120820251600/...train_folder... \
    --beh_test_rel  120820251600/...test_folder... \
    --beh_val_rel   120820251600/...val_folder... \
    --target_eval_rel 130820251600/...target_folder... \
    --output_dir ${{outputs.model}} \
    --device auto \
    --amp
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

import ray
from ray.rllib.offline.json_reader import JsonReader

# Your helper utils (must be available in the code folder / image)
from oppe_utils import (
    load_checkpoint,
    calculate_policy_expected_value,
    load_json_to_df_max,
)

# Optional debugging (won't fail if debugpy isn't installed in the image)
try:
    import debugpy  # type: ignore
except Exception:  # pragma: no cover
    debugpy = None


# Reproducibility (optional)
torch.manual_seed(0)
np.random.seed(0)


def _get_device(device_arg: str) -> torch.device:
    device_arg = (device_arg or "auto").lower()
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _optimizer_to(optimizer: optim.Optimizer, device: torch.device) -> None:
    """Move optimizer internal state tensors to device (important when resuming)."""
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _to_numpy_obs(x: torch.Tensor) -> np.ndarray:
    """RLlib policies typically expect numpy on CPU."""
    return x.detach().cpu().numpy()


def _compute_actions_batch(policy, obs_batch: torch.Tensor) -> torch.Tensor:
    """
    Try fast batch action computation via RLlib policy.compute_actions().
    Fall back to per-sample compute_single_action() if needed.
    Returns actions as int64 torch tensor on CPU.
    """
    obs_np = _to_numpy_obs(obs_batch)

    # Normalize common shapes: (B, 1, D) -> (B, D)
    if obs_np.ndim >= 3:
        obs_np = obs_np.reshape(obs_np.shape[0], -1)

    if hasattr(policy, "compute_actions"):
        try:
            acts, _states, _infos = policy.compute_actions(obs_np, explore=False)
            return torch.as_tensor(acts, dtype=torch.int64)
        except Exception:
            pass

    acts_list: List[int] = []
    for i in range(obs_batch.shape[0]):
        a, _s, _info = policy.compute_single_action(_to_numpy_obs(obs_batch[i]), explore=False)
        acts_list.append(int(a))
    return torch.tensor(acts_list, dtype=torch.int64)


# == Q-network for FQE ==
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def _infer_state_dim(states: torch.Tensor) -> int:
    """
    states may be shaped:
      - (N, D)
      - (N, 1, D)  (common when obs are stored as [1, D])
      - (N, ..., D) -> we use last dim
    """
    if states.ndim >= 2:
        return int(states.shape[-1])
    return 1


def _find_latest_checkpoint(save_dir: Path) -> Optional[Path]:
    ckpts = sorted(save_dir.glob("fqe_epoch_*.pt"))
    if not ckpts:
        return None

    def epoch_num(p: Path) -> int:
        import re

        m = re.search(r"fqe_epoch_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    ckpts = sorted(ckpts, key=epoch_num)
    return ckpts[-1]


def _state_dict_to_cpu(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Make checkpoints portable (smaller GPU dependency) by saving tensors on CPU."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        out[k] = v.detach().cpu() if torch.is_tensor(v) else v
    return out


def train_nn(
    df,
    policy_action,
    save_dir: Path,
    resume_training: bool = True,
    num_epochs: int = 200,
    batch_size: int = 4096,
    target_update_interval: int = 10,
    save_every: int = 20,
    lr: float = 5e-4,
    gamma: float = 0.99,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = False,
    val_split: float = 0.2,
    early_stopping_patience: int = 30,
    min_delta: float = 0.8,
) -> Tuple[QNetwork, Dict[str, Any]]:
    """
    Trains an FQE-style Q network.

    Returns:
      - trained q_net (on 'device')
      - metadata dict (state_dim, num_actions, gamma, etc.)
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Keep dataset tensors on CPU; move batches to GPU per-iteration to control VRAM.
    states = torch.tensor(np.stack(df["obs"].values), dtype=torch.float32, device=torch.device("cpu"))
    actions = torch.tensor(df["action"].values, dtype=torch.int64, device=torch.device("cpu"))
    rewards = torch.tensor(df["reward"].values, dtype=torch.float32, device=torch.device("cpu"))
    next_states = torch.tensor(np.stack(df["next_state"].values), dtype=torch.float32, device=torch.device("cpu"))
    dones = torch.tensor(df["done"].values, dtype=torch.float32, device=torch.device("cpu"))

    state_dim = _infer_state_dim(states)
    num_actions = int(actions.max().item() + 1)
    n_samples = states.shape[0]

    if n_samples < 2:
        raise ValueError(f"[FQE] Need at least 2 samples for split, got {n_samples}.")

    n_val = max(1, int(n_samples * val_split))
    n_train = n_samples - n_val
    if n_train < 1:
        n_train = 1
        n_val = n_samples - 1

    print(f"[FQE] state_dim={state_dim} num_actions={num_actions}")
    print(f"[FQE] dataset transitions={states.shape[0]} batch_size={batch_size} num_epochs={num_epochs}")
    print(f"[FQE] split train={n_train} val={n_val} (val_split={val_split:.2f})")

    # Fixed split for stable early-stopping metric across epochs.
    perm = torch.randperm(n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    q_net = QNetwork(state_dim, num_actions).to(device)
    target_net = QNetwork(state_dim, num_actions).to(device)
    print("[FQE] q_net param device:", next(q_net.parameters()).device)
    print("[FQE] target_net param device:", next(target_net.parameters()).device)

    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))

    # Resume from latest checkpoint (if any)
    start_epoch = 0
    if resume_training:
        latest = _find_latest_checkpoint(save_dir)
        if latest is not None:
            checkpoint = torch.load(latest, map_location="cpu")
            q_net.load_state_dict(checkpoint["model_state_dict"])
            target_net.load_state_dict(checkpoint["target_model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            q_net.to(device)
            target_net.to(device)
            _optimizer_to(optimizer, device)

            start_epoch = int(checkpoint["epoch"])
            print(f"[FQE] Resuming from {latest} (start_epoch={start_epoch})")
        else:
            print("[FQE] No checkpoint found, training from scratch.")
    else:
        print("[FQE] resume_training=False, training from scratch.")

    best_val_loss = float("inf")
    patience_counter = 0

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        train_shuffle = train_idx[torch.randperm(train_idx.shape[0])]
        batch_losses: List[float] = []

        q_net.train()
        for i in range(0, train_shuffle.shape[0], batch_size):
            batch_idx = train_shuffle[i : i + batch_size]

            # Move batch to device
            batch_states = states[batch_idx].to(device, non_blocking=True)
            batch_actions = actions[batch_idx].to(device, non_blocking=True)
            batch_rewards = rewards[batch_idx].to(device, non_blocking=True)
            batch_next_states = next_states[batch_idx].to(device, non_blocking=True)
            batch_dones = dones[batch_idx].to(device, non_blocking=True)

            # Forward: Q(s, :)
            with autocast(enabled=(use_amp and device.type == "cuda")):
                q_values = q_net(batch_states).squeeze(1)
                q_sa = q_values.gather(1, batch_actions.view(-1, 1)).squeeze(1)

            # Target: r + gamma * Q_target(s', a'(s'))
            with torch.no_grad():
                # RLlib policy inference expects CPU numpy typically
                next_actions_cpu = _compute_actions_batch(policy_action, batch_next_states)
                next_actions = next_actions_cpu.to(device, non_blocking=True)

                q_next = target_net(batch_next_states).squeeze(1)
                q_next_sa = q_next.gather(1, next_actions.view(-1, 1)).squeeze(1)
                q_next_sa = q_next_sa * (1 - batch_dones)

                target_values = batch_rewards + gamma * q_next_sa

            with autocast(enabled=(use_amp and device.type == "cuda")):
                loss = criterion(q_sa, target_values)

            batch_losses.append(float(loss.item()))

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # Target update
        if (epoch + 1) % target_update_interval == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Checkpoint
        if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                "epoch": epoch + 1,
                # save on CPU for portability
                "model_state_dict": _state_dict_to_cpu(q_net.state_dict()),
                "target_model_state_dict": _state_dict_to_cpu(target_net.state_dict()),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": float(np.mean(batch_losses)) if batch_losses else None,
                "state_dim": state_dim,
                "num_actions": num_actions,
                "gamma": gamma,
                "lr": lr,
                "amp": bool(use_amp and device.type == "cuda"),
                "device": str(device),
            }
            checkpoint_path = save_dir / f"fqe_epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"[FQE] Saved checkpoint: {checkpoint_path}")

        q_net.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for j in range(0, val_idx.shape[0], batch_size):
                v_idx = val_idx[j : j + batch_size]
                v_states = states[v_idx].to(device, non_blocking=True)
                v_actions = actions[v_idx].to(device, non_blocking=True)
                v_rewards = rewards[v_idx].to(device, non_blocking=True)
                v_next_states = next_states[v_idx].to(device, non_blocking=True)
                v_dones = dones[v_idx].to(device, non_blocking=True)

                with autocast(enabled=(use_amp and device.type == "cuda")):
                    v_q_values = q_net(v_states).squeeze(1)
                    v_q_sa = v_q_values.gather(1, v_actions.view(-1, 1)).squeeze(1)

                next_actions_cpu = _compute_actions_batch(policy_action, v_next_states)
                next_actions = next_actions_cpu.to(device, non_blocking=True)
                v_q_next = target_net(v_next_states).squeeze(1)
                v_q_next_sa = v_q_next.gather(1, next_actions.view(-1, 1)).squeeze(1)
                v_q_next_sa = v_q_next_sa * (1 - v_dones)
                v_targets = v_rewards + gamma * v_q_next_sa

                with autocast(enabled=(use_amp and device.type == "cuda")):
                    v_loss = criterion(v_q_sa, v_targets)

                val_losses.append(float(v_loss.item()))

        avg_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(
            f"[FQE] Epoch {epoch+1}/{num_epochs} train_loss={avg_loss:.6f} "
            f"val_loss={avg_val_loss:.6f} patience={patience_counter}/{early_stopping_patience}"
        )

        # Save best model by validation loss
        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(_state_dict_to_cpu(q_net.state_dict()), save_dir / "best_model.pt")
            print(f"[FQE] New best model (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(
                    f"[FQE] Early stopping at epoch {epoch+1}. "
                    f"Best val_loss={best_val_loss:.6f}"
                )
                break

    best_model_path = save_dir / "best_model.pt"
    if best_model_path.exists():
        best_sd = torch.load(best_model_path, map_location="cpu")
        q_net.load_state_dict(best_sd)
        q_net.to(device)
        print(f"[FQE] Loaded best model from: {best_model_path}")

    meta = {
        "state_dim": state_dim,
        "num_actions": num_actions,
        "gamma": gamma,
        "lr": lr,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "target_update_interval": target_update_interval,
        "save_every": save_every,
        "device": str(device),
        "amp": bool(use_amp and device.type == "cuda"),
        "val_split": val_split,
        "train_samples": int(n_train),
        "val_samples": int(n_val),
        "best_val_loss": float(best_val_loss),
        "early_stopping_patience": early_stopping_patience,
        "min_delta": min_delta,
    }
    return q_net, meta


def evaluate_policy(q_net: QNetwork, df, policy_action) -> Optional[float]:
    device = next(q_net.parameters()).device

    # Keep obs on CPU for RLlib policy inference, then move to model device for Q eval
    states_cpu = torch.tensor(np.stack(df["obs"].values), dtype=torch.float32, device=torch.device("cpu"))

    q_net.eval()
    initial_state_indices = []
    N = len(df)
    for i in range(N):
        if i == 0 or df.iloc[i - 1]["done"]:
            initial_state_indices.append(i)

    if not initial_state_indices:
        print("[Eval] No initial states found (check df ordering / done flags).")
        return None

    init_states_cpu = states_cpu[initial_state_indices]
    actions_cpu = _compute_actions_batch(policy_action, init_states_cpu)

    init_states = init_states_cpu.to(device, non_blocking=True)
    actions = actions_cpu.to(device, non_blocking=True)

    with torch.no_grad():
        q_vals = q_net(init_states).squeeze(1)  # (K,A) or (K,1,A)->(K,A)
        q_sa = q_vals.gather(1, actions.view(-1, 1)).squeeze(1)
        values = q_sa.detach().cpu().numpy()

    estimated_value = float(np.mean(values))
    print(f"[Eval] Estimated expected value from initial states: {estimated_value:.6f}")
    return estimated_value


def _resolve_dir(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    return p


def run_oppe(args: argparse.Namespace) -> None:
    device = _get_device(args.device)

    # Optional Torch perf knobs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            # torch>=2.0: improves matmul perf in some cases
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print(f"[Torch] version={torch.__version__} cuda_available={torch.cuda.is_available()} device={device}")
    if torch.cuda.is_available():
        print(f"[Torch] cuda_device_count={torch.cuda.device_count()}")
        try:
            print(f"[Torch] cuda_device_name={torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    episodes_root = Path(args.episodes_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Put FQE checkpoints inside output so AML uploads them
    save_dir = output_dir / args.checkpoints_subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resolve episode dirs
    beh_train_dir = _resolve_dir(episodes_root, args.beh_train_rel)
    beh_test_dir = _resolve_dir(episodes_root, args.beh_test_rel)
    beh_val_dir = _resolve_dir(episodes_root, args.beh_val_rel)
    target_eval_dir = _resolve_dir(episodes_root, args.target_eval_rel)

    print("[Paths] episodes_root:", episodes_root)
    print("[Paths] beh_train_dir:", beh_train_dir)
    print("[Paths] beh_test_dir :", beh_test_dir)
    print("[Paths] beh_val_dir  :", beh_val_dir)
    print("[Paths] target_eval_dir:", target_eval_dir)
    print("[Paths] beh_checkpoint_dir:", args.beh_checkpoint_dir)
    print("[Paths] eval_checkpoint_dir:", args.eval_checkpoint_dir)
    print("[Paths] output_dir:", output_dir)
    print("[Paths] save_dir (checkpoints):", save_dir)

    # Load target policy (RLlib) checkpoint
    eval_agent = load_checkpoint(args.eval_checkpoint_dir)

    # Readers
    reader_beh_val = JsonReader(str(beh_val_dir))
    reader_beh_train = JsonReader(str(beh_train_dir))
    reader_beh_test = JsonReader(str(beh_test_dir))
    reader_target = JsonReader(str(target_eval_dir))

    # ---------------------------------------------------------------------------#
    # 1) "Real" expected value from recorded episodes (baseline check)
    # ---------------------------------------------------------------------------#
    beh_eps_val_df, eps, steps = load_json_to_df_max(reader_beh_val)
    print(f"[Data] loaded beh_val episodes: eps={eps} steps={steps}")

    target_eps_df, eps_t, steps_t = load_json_to_df_max(reader_target)
    print(f"[Data] loaded target_eval episodes: eps={eps_t} steps={steps_t}")

    beh_expected_return, beh_return_stdev = calculate_policy_expected_value(beh_eps_val_df, args.gamma)
    target_expected_return, target_return_stdev = calculate_policy_expected_value(target_eps_df, args.gamma)

    print(f"[Returns] BEH_POLICY   mean={beh_expected_return:.6f} std={beh_return_stdev:.6f}")
    print(f"[Returns] TARGET_POLICY mean={target_expected_return:.6f} std={target_return_stdev:.6f}")

    # ---------------------------------------------------------------------------#
    # 2) Train DM estimator (FQE Q-model)
    # ---------------------------------------------------------------------------#
    print("\n[FQE] Training Q-model...")
    beh_train_df, eps_train, steps_train = load_json_to_df_max(reader_beh_train, args.beh_train_limit)
    print(f"[Data] loaded beh_train episodes: eps={eps_train} steps={steps_train} limit={args.beh_train_limit}")

    q_net, meta = train_nn(
        beh_train_df,
        eval_agent,
        save_dir=save_dir,
        resume_training=args.resume_training,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
        save_every=args.save_every,
        lr=args.lr,
        gamma=args.gamma,
        device=device,
        use_amp=args.amp,
    )

    # Evaluate on test
    beh_test_df, eps_test, steps_test = load_json_to_df_max(reader_beh_test, args.beh_test_limit)
    print(f"[Data] loaded beh_test episodes: eps={eps_test} steps={steps_test} limit={args.beh_test_limit}")
    est_value = evaluate_policy(q_net, beh_test_df, eval_agent)

    # Save final artifacts (AML will upload output_dir)
    final_path = output_dir / "fqe_final.pt"
    torch.save(
        {
            "model_state_dict": _state_dict_to_cpu(q_net.state_dict()),
            "meta": meta,
            "estimated_value_test": est_value,
        },
        final_path,
    )
    print(f"[FQE] Saved final model to: {final_path}")

    # Save metadata as json
    meta_path = output_dir / "fqe_meta.json"
    try:
        import json

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    **meta,
                    "estimated_value_test": est_value,
                    "beh_expected_return_val": beh_expected_return,
                    "target_expected_return_eval": target_expected_return,
                    "beh_val_std": beh_return_stdev,
                    "target_eval_std": target_return_stdev,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"[FQE] Saved metadata to: {meta_path}")
    except Exception as e:
        print(f"[Warn] Could not write meta json: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # AML inputs/outputs
    p.add_argument(
        "--episodes_root",
        type=str,
        required=True,
        help="AML uri_folder mount/download root containing JSON episode folders.",
    )
    p.add_argument("--output_dir", type=str, required=True, help="AML output folder (e.g., ${{outputs.model}}).")

    # Checkpoints for policies (RLlib)
    p.add_argument("--beh_checkpoint_dir", type=str, required=False, default="", help="(Optional) Behavior policy checkpoint dir if needed.")
    p.add_argument("--eval_checkpoint_dir", type=str, required=True, help="Target/eval policy checkpoint dir (RLlib).")

    # Episode subfolders inside episodes_root
    p.add_argument("--beh_train_rel", type=str, required=True, help="Relative path under episodes_root to behavior TRAIN episodes folder.")
    p.add_argument("--beh_test_rel", type=str, required=True, help="Relative path under episodes_root to behavior TEST episodes folder.")
    p.add_argument("--beh_val_rel", type=str, required=True, help="Relative path under episodes_root to behavior VAL episodes folder.")
    p.add_argument("--target_eval_rel", type=str, required=True, help="Relative path under episodes_root to target/eval episodes folder.")

    # Training limits / hyperparams
    p.add_argument("--beh_train_limit", type=int, default=10000, help="Max transitions to load for training (passed to load_json_to_df_max).")
    p.add_argument("--beh_test_limit", type=int, default=2000, help="Max transitions to load for test (passed to load_json_to_df_max).")
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--target_update_interval", type=int, default=20)
    p.add_argument("--save_every", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)

    # Persistence / resume
    p.add_argument("--checkpoints_subdir", type=str, default="fqe_checkpoints", help="Subdir inside output_dir to store checkpoints.")
    p.add_argument("--resume_training", action="store_true", help="Resume from latest checkpoint if present in checkpoints_subdir.")
    p.add_argument("--no_resume_training", dest="resume_training", action="store_false")
    p.set_defaults(resume_training=True)

    # Device / GPU
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device. auto=use CUDA if available.")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only).")

    # Debugging
    p.add_argument("--debug", action="store_true", help="Enable debugpy wait_for_client on port 5678 (if debugpy installed).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.debug and debugpy is not None:
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for VS Code debugger attach on port 5678...")
        debugpy.wait_for_client()

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    print("[Ray] version:", ray.__version__)

    run_oppe(args)


if __name__ == "__main__":
    main()
