# pip install "ray[rllib]==2.11.0" scope-rl d3rlpy gymnasium numpy

import json
import numpy as np
import pandas as pd
import debugpy
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Sequence, Union, Tuple, Any, Iterable
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from oppe_utils import load_json_to_df

import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.offline.json_reader import JsonReader

from scope_rl.ope import CreateOPEInput, OffPolicyEvaluation
from scope_rl.ope.discrete import DirectMethod as DM, SelfNormalizedPDIS as SNPDIS, SelfNormalizedDR as SNDR
from scope_rl.policy.head import BaseHead
from scope_rl.dataset import SyntheticDataset

debug = 1
if debug == 1:
    # Escucha en el puerto 5678 (puedes cambiarlo)
    debugpy.listen(("0.0.0.0", 5678))
    print("Esperando debugger de VS Code para conectar...")
    debugpy.wait_for_client()

# ----------------------------
# 1) Loader: RLlib JSON -> logged_dataset (lista de episodios)
# ----------------------------
def _as_np(x):
    return np.asarray(x) if x is not None else None

def load_rllib_logged_dataset(path: str) -> List[Dict[str, np.ndarray]]:
    """
    Devuelve una lista de episodios con claves:
      'state', 'action', 'reward', 'next_state', 'done', (opcional) 'pscore'
    Soporta:
      A) JSON Lines por paso con 'episode_id', 'obs', 'new_obs'/'next_obs', ...
      B) Dict con 'batches': list[SampleBatch] con claves 'obs', 'actions', ...
      C) Lista de episodios con 'observations', 'actions', ...
    """
    def try_parse_json_lines():
        episodes = defaultdict(lambda: defaultdict(list))
        with open(path, "r") as f:
            any_line = False
            for line in f:
                line = line.strip()
                if not line:
                    continue
                any_line = True
                rec = json.loads(line)
                # Campos comunes en RLlib por paso
                eid = rec.get("episode_id") or rec.get("eid") or 0
                obs = rec.get("obs") or rec.get("state") or rec.get("observation")
                nxt = rec.get("new_obs") or rec.get("next_obs") or rec.get("next_observation")
                act = rec.get("actions") if isinstance(rec.get("actions"), (int, float, list)) else rec.get("action")
                rew = rec.get("rewards") if isinstance(rec.get("rewards"), (int, float, list)) else rec.get("reward")
                done = rec.get("dones") if isinstance(rec.get("dones"), (bool, list, int)) else rec.get("done")
                psc  = rec.get("pscore") or rec.get("action_prob") or rec.get("action_probabilities")

                episodes[eid]["state"].append(obs)
                if nxt is not None:
                    episodes[eid]["next_state"].append(nxt)
                episodes[eid]["action"].append(act)
                episodes[eid]["reward"].append(rew)
                episodes[eid]["done"].append(done)
                if psc is not None:
                    episodes[eid]["pscore"].append(psc)

            if not any_line:
                return None  # no era JSONL
        # Cerrar episodios: asegurar next_state por desplazamiento si no llegó
        out = []
        for eid, buf in episodes.items():
            S = _as_np(buf["state"])
            A = _as_np(buf["action"])
            R = _as_np(buf["reward"])
            D = _as_np(buf["done"]).astype(bool)
            if "next_state" in buf and len(buf["next_state"]) == len(S):
                NS = _as_np(buf["next_state"])
            else:
                # Derivar next_state por desplazamiento si es posible
                NS = np.vstack([S[1:], S[-1:]]) if S.ndim >= 1 else S
            ep = {"state": S, "action": A, "reward": R, "next_state": NS, "done": D}
            if "pscore" in buf:
                P = _as_np(buf["pscore"])
                # Si pscore es vector por paso, usarlo tal cual; si viene como probs(a|s) vectoriales, extraer si procede.
                ep["pscore"] = P
            out.append(ep)
        return out

    def try_parse_batches_dict(obj: Dict[str, Any]):
        if not isinstance(obj, dict) or "batches" not in obj:
            return None
        out = []
        for b in obj["batches"]:
            # Esperado en SampleBatch: 'obs','actions','rewards','dones','new_obs'
            S  = _as_np(b.get("obs"))
            A  = _as_np(b.get("actions"))
            R  = _as_np(b.get("rewards"))
            D  = _as_np(b.get("dones")).astype(bool)
            NS = _as_np(b.get("new_obs") or b.get("next_obs"))
            if NS is None and S is not None and len(S) == len(A):
                NS = np.vstack([S[1:], S[-1:]]) if S.ndim >= 1 else S
            ep = {"state": S, "action": A, "reward": R, "next_state": NS, "done": D}
            if "pscore" in b or "action_prob" in b:
                ep["pscore"] = _as_np(b.get("pscore") or b.get("action_prob"))
            out.append(ep)
        return out

    def try_parse_episode_list(obj):
        if not isinstance(obj, list):
            return None
        out = []
        ok = False
        for ep in obj:
            # Formato tipo RolloutSaver: 'observations','actions','rewards','dones','next_observations'
            if any(k in ep for k in ("observations", "obs", "state")):
                ok = True
                S  = _as_np(ep.get("observations") or ep.get("obs") or ep.get("state"))
                A  = _as_np(ep.get("actions") or ep.get("action"))
                R  = _as_np(ep.get("rewards") or ep.get("reward"))
                D  = _as_np(ep.get("dones") or ep.get("done")).astype(bool)
                NS = _as_np(ep.get("next_observations") or ep.get("new_obs") or ep.get("next_state"))
                if NS is None and S is not None and len(S) == len(A):
                    NS = np.vstack([S[1:], S[-1:]]) if S.ndim >= 1 else S
                item = {"state": S, "action": A, "reward": R, "next_state": NS, "done": D}
                if "pscore" in ep or "action_prob" in ep:
                    item["pscore"] = _as_np(ep.get("pscore") or ep.get("action_prob"))
                out.append(item)
        return out if ok else None

    # 1) Intento JSON Lines
    episodes = try_parse_json_lines()
    if episodes is not None:
       return episodes

    # 2) Carga JSON normal
    with open(path, "r") as f:
        obj = json.load(f)

    # 2a) batches dict
    episodes = try_parse_batches_dict(obj)
    if episodes is not None:
        return episodes

    # 2b) lista de episodios
    # episodes = try_parse_episode_list(obj)
    # if episodes is not None:
    #    return episodes

    raise ValueError("Formato de JSON RLlib no reconocido. Asegúrate de que contenga JSONL por paso, "
                     "un dict con 'batches', o una lista de episodios.")


import numpy as np
import pandas as pd
from typing import Dict, Optional, Sequence, Union

def df_to_logged_dataset(
    df: pd.DataFrame,
    *,
    ep_col: str = "ep",
    step_col: str = "step",
    obs_col: str = "obs",
    next_obs_col: str = "next_state",
    action_col: str = "action",
    reward_col: str = "reward",
    done_col: str = "done",
    truncated_col: Optional[str] = None,        # si tienes "truncated" (terminación por límite de pasos)
    pscore_col: Optional[str] = "action_prob",  # prob(a_t | s_t)
    alt_logp_col: Optional[str] = "action_logp",
    behavior_policy: str = "behavior_policy",
    dataset_id: int = 0,
    # metadatos / formato
    action_type: Optional[str] = None,          # {"discrete","continuous"} o None para inferir
    n_actions: Optional[int] = None,            # si discrete; intenta inferir si None
    one_hot_discrete: bool = False,             # para obtener acción (size, n_actions) en discreto
    action_keys: Optional[Sequence[str]] = None,
    action_meaning: Optional[Dict[int, Union[int,str]]] = None,
    state_keys: Optional[Sequence[str]] = None,
    float_dtype = np.float32,
    eps: float = 1e-12,
    # NUEVO: filtro de longitud mínima de episodio
    min_steps: int = 200,
) -> Dict[str, Union[int, str, np.ndarray, Dict]]:
    """Convierte un DataFrame (una fila por (episodio, paso)) al esquema de logged_dataset de SCOPE-RL.
    
    Solo incluye episodios con longitud >= min_steps (por defecto, 200).
    """
    required = {ep_col, step_col, obs_col, action_col, reward_col, done_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    # Orden estable por (ep, step)
    df_sorted = df.sort_values([ep_col, step_col], kind="mergesort")

    # Utilidades de apilado robusto
    def to_arr_list(series: pd.Series):
        return [np.asarray(v) for v in series.to_list()]

    def stack_consistent(arrs, name):
        # Escalares -> (T,)
        if all(a.ndim == 0 for a in arrs):
            return np.asarray(arrs, dtype=float_dtype)
        # 1D homogéneo -> (T, d)
        if all(a.ndim == 1 for a in arrs):
            shapes = {a.shape for a in arrs}
            if len(shapes) != 1:
                raise ValueError(f"{name}: formas 1D inconsistentes: {shapes}")
            return np.stack(arrs).astype(float_dtype)
        # kD homogéneo -> (T, ...)
        shapes = {a.shape for a in arrs}
        if len(shapes) == 1:
            return np.stack(arrs).astype(float_dtype)
        raise ValueError(f"{name}: mezcla de formas incompatible: {[(i,a.shape) for i,a in enumerate(arrs)]}")

    # Recoger por episodios (filtrando los cortos)
    episodes_S, episodes_A, episodes_R, episodes_done = [], [], [], []
    episodes_trunc = []
    episodes_len = []

    pscore_list = []

    # Inferir action_type si no se indica
    inferred_action_type = None

    for ep, g in df_sorted.groupby(ep_col, sort=False):
        # Filtrado por longitud mínima del episodio
        if len(g) < min_steps:
            continue

        S = stack_consistent(to_arr_list(g[obs_col]), "state")
        S = S.squeeze(1)
        A_list = to_arr_list(g[action_col])

        # Inferir tipo de acción por episodio si no viene dado
        if action_type is None:
            if all(a.ndim == 0 for a in A_list):
                inferred_action_type = inferred_action_type or "discrete"
            else:
                inferred_action_type = inferred_action_type or "continuous"

        # Apilar acción con formato provisional
        if all(a.ndim == 0 for a in A_list):
            A = np.asarray(A_list)  # (T,)
        else:
            A = stack_consistent(A_list, "action")  # (T, d_a)

        R = np.asarray(g[reward_col].to_list(), dtype=float_dtype)
        D = np.asarray(g[done_col].to_list(), dtype=bool).copy()
        if D.size > 0:
            D[-1] = True  # fuerza que el episodio termine, como estamos truncando, sería casualidad que 
                          # el episodio acabara en el paso 200...

        # terminal: True en el último paso SIEMPRE (marca el corte de trayectoria).
        # 'done' distinguirá terminación ambiental vs timeout.
        Tm = np.zeros_like(D, dtype=bool)
        if len(Tm) > 0:
            Tm[-1] = True

        # Si tienes una columna de "truncated", úsala para derivar timeouts externamente si lo necesitas,
        # pero aquí mantenemos "terminal" como bandera de fin de trayectoria.
        # (SCOPE-RL suele derivar timeouts = terminal & ~done internamente)

        # pscore episodio
        if pscore_col and pscore_col in g.columns:
            P = np.asarray(g[pscore_col].to_list(), dtype=float_dtype)
        elif alt_logp_col and alt_logp_col in g.columns:
            P = np.exp(np.asarray(g[alt_logp_col].to_list(), dtype=float_dtype))
        else:
            P = None
        if P is not None:
            P = np.clip(P, eps, 1.0).astype(float_dtype)
            if np.any(~np.isfinite(P)):
                raise ValueError(f"Episodio {ep}: pscore contiene NaN/Inf")

        episodes_S.append(S)
        episodes_A.append(A)
        episodes_R.append(R)
        episodes_done.append(D)
        episodes_trunc.append(Tm)
        episodes_len.append(len(R))
        pscore_list.append(P)

    # Si no queda ningún episodio tras el filtro, devolvemos un dataset vacío consistente
    if len(episodes_len) == 0:
        return {
            "size": 0,
            "n_trajectories": 0,
            "step_per_trajectory": None,
            "action_type": (action_type or "discrete"),
            "n_actions": None if (action_type or "discrete") != "discrete" else None,
            "action_dim": None if (action_type or "discrete") == "discrete" else None,
            "action_keys": list(action_keys) if action_keys is not None else None,
            "action_meaning": dict(action_meaning) if action_meaning is not None else None,
            "state_dim": None,
            "state_keys": list(state_keys) if state_keys is not None else None,
            "state": np.array([], dtype=float_dtype).reshape(0, 0),
            "action": np.array([], dtype=float_dtype),
            "reward": np.array([], dtype=float_dtype),
            "done": np.array([], dtype=bool),
            "terminal": np.array([], dtype=bool),
            "info": None,
            "pscore": None,
            "behavior_policy": behavior_policy,
            "dataset_id": int(dataset_id),
        }

    # Determinar action_type final
    action_type_final = action_type or (inferred_action_type or "discrete")

    # Comprobar si hay longitud fija (para step_per_trajectory)
    unique_lengths = sorted(set(episodes_len))
    if len(unique_lengths) == 1:
        step_per_trajectory = int(unique_lengths[0])
    else:
        step_per_trajectory = None  # longitudes variables; dejamos None y confiamos en done/terminal

    n_trajectories = len(episodes_len)
    size = int(np.sum(episodes_len))

    # Construir tensores aplanados
    def flatten_list_of_arrays(lst):
        return np.concatenate(lst, axis=0) if len(lst) else np.array([], dtype=float_dtype)

    # Estado: normalizamos a 2D (size, d_s)
    S_flat = flatten_list_of_arrays([s if s.ndim >= 1 else s.reshape(-1, 1) for s in episodes_S])
    if S_flat.ndim == 1:
        S_flat = S_flat.reshape(-1, 1)
    state_dim = int(S_flat.shape[1]) if S_flat.size else None

    # Acción
    if action_type_final == "discrete":
        if any(a.ndim > 1 for a in episodes_A):
            raise ValueError("Se infirió 'discrete' pero hay acciones vectoriales; fija action_type='continuous'.")
        A_flat_int = flatten_list_of_arrays([a.astype(int) for a in episodes_A])  # (size,)
        if n_actions is None:
            n_actions = int(A_flat_int.max()) + 1 if A_flat_int.size else None

        if one_hot_discrete and n_actions is not None:
            A_flat = np.eye(n_actions, dtype=float_dtype)[A_flat_int]
        else:
            A_flat = A_flat_int  # (size,)
        action_dim = None
    else:
        # continuous
        A_flat = flatten_list_of_arrays([a if a.ndim >= 1 else a.reshape(-1, 1) for a in episodes_A])
        if A_flat.ndim == 1:
            A_flat = A_flat.reshape(-1, 1)
        action_dim = int(A_flat.shape[1]) if A_flat.size else None
        n_actions = None
    A_flat_torch = torch.from_numpy(A_flat)
    R_flat = flatten_list_of_arrays(episodes_R).astype(float_dtype)
    done_flat = flatten_list_of_arrays([d.astype(bool) for d in episodes_done]).astype(bool)
    term_flat = flatten_list_of_arrays([t.astype(bool) for t in episodes_trunc]).astype(bool)

    # pscore
    if any(p is not None for p in pscore_list):
        P_list = []
        for T, P in zip(episodes_len, pscore_list):
            if P is None:
                P = np.full(T, np.nan, dtype=float_dtype)
            P_list.append(P)
        pscore_flat = np.concatenate(P_list).astype(float_dtype)
    else:
        pscore_flat = None

    # Claves opcionales
    action_keys = list(action_keys) if action_keys is not None else None
    state_keys = list(state_keys) if state_keys is not None else None
    action_meaning = dict(action_meaning) if action_meaning is not None else None

    # Ensamblar diccionario final (siguiendo el orden/documentación)
    logged_dataset: Dict[str, Union[int, str, np.ndarray, Dict]] = {
        "size": size,
        "n_trajectories": n_trajectories,
        "step_per_trajectory": step_per_trajectory,   # puede ser None si longitudes variables
        "action_type": action_type_final,             # "discrete" | "continuous"
        "n_actions": n_actions,                       # solo para discreto
        "action_dim": action_dim,                     # solo para continuo
        "action_keys": action_keys,                   # opcional
        "action_meaning": action_meaning,             # opcional
        "state_dim": state_dim,
        "state_keys": state_keys,                     # opcional
        "state": S_flat,                              # (size, d_s)
        "action": A_flat,                             # (size,) o (size,n_actions)/(size,d_a)
        "reward": R_flat,                             # (size,)
        "done": done_flat,                            # (size,)
        "terminal": term_flat,                        # (size,)
        "info": None,                                 # opcional: dict si lo tienes
        "pscore": pscore_flat,                        # (size,) con NaN donde no haya
        "behavior_policy": behavior_policy,
        "dataset_id": int(dataset_id),
    }

    return logged_dataset

def df_to_logged_dataset_torch(
    df: pd.DataFrame,
    *,
    ep_col: str = "ep",
    step_col: str = "step",
    obs_col: str = "obs",
    next_obs_col: str = "next_state",
    action_col: str = "action",
    reward_col: str = "reward",
    done_col: str = "done",
    truncated_col: Optional[str] = None,        # si tienes "truncated" (terminación por límite de pasos)
    pscore_col: Optional[str] = "action_prob",  # prob(a_t | s_t)
    alt_logp_col: Optional[str] = "action_logp",
    behavior_policy: str = "behavior_policy",
    dataset_id: int = 0,
    # metadatos / formato
    action_type: Optional[str] = None,          # {"discrete","continuous"} o None para inferir
    n_actions: Optional[int] = None,            # si discrete; intenta inferir si None
    one_hot_discrete: bool = False,             # para obtener acción (size, n_actions) en discreto
    action_keys: Optional[Sequence[str]] = None,
    action_meaning: Optional[Dict[int, Union[int,str]]] = None,
    state_keys: Optional[Sequence[str]] = None,
    float_dtype: torch.dtype = torch.float32,
    eps: float = 1e-12,
    # NUEVO: filtro de longitud mínima de episodio
    min_steps: int = 200,
    # dispositivo
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Union[int, str, torch.Tensor, Dict]]:
    """
    Convierte un DataFrame (una fila por (episodio, paso)) al esquema de logged_dataset para SCOPE-RL,
    devolviendo TENSORES PyTorch. Solo incluye episodios con longitud >= min_steps.
    """

    required = {ep_col, step_col, obs_col, action_col, reward_col, done_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    # Orden estable por (ep, step)
    df_sorted = df.sort_values([ep_col, step_col], kind="mergesort")

    # ---------- utilidades Torch ----------
    def to_tensor_list(series: pd.Series, *, dtype: Optional[torch.dtype] = None) -> Sequence[torch.Tensor]:
        # Cada elemento puede ser escalar, lista, np.array, torch.Tensor...
        out = []
        for v in series.to_list():
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            if dtype is not None and t.dtype != dtype and t.is_floating_point():
                t = t.to(dtype)
            out.append(t.to(device))
        return out

    def stack_consistent(tensors: Sequence[torch.Tensor], name: str, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # Escalares -> (T,)
        if all(t.ndim == 0 for t in tensors):
            tcat = torch.stack(tensors).to(device)
            if dtype is not None and tcat.is_floating_point():
                tcat = tcat.to(dtype)
            return tcat
        # 1D homogéneo -> (T, d)
        if all(t.ndim == 1 for t in tensors):
            shapes = {tuple(t.shape) for t in tensors}
            if len(shapes) != 1:
                raise ValueError(f"{name}: formas 1D inconsistentes: {shapes}")
            tcat = torch.stack(tensors, dim=0).to(device)
            if dtype is not None and tcat.is_floating_point():
                tcat = tcat.to(dtype)
            return tcat
        # kD homogéneo -> (T, ...)
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) == 1:
            tcat = torch.stack(tensors, dim=0).to(device)
            if dtype is not None and tcat.is_floating_point():
                tcat = tcat.to(dtype)
            return tcat
        raise ValueError(f"{name}: mezcla de formas incompatible: {[(i,tuple(a.shape)) for i,a in enumerate(tensors)]}")

    def flatten_list_of_tensors(lst: Sequence[torch.Tensor], *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if len(lst) == 0:
            return torch.empty(0, device=device, dtype=(dtype or float_dtype))
        if len(lst) == 1:
            out = lst[0]
        else:
            out = torch.cat(lst, dim=0)
        if dtype is not None and out.is_floating_point():
            out = out.to(dtype)
        return out

    def flatten_list_of_arrays(lst):
        return np.concatenate(lst, axis=0) if len(lst) else np.array([], dtype=float_dtype)

    # ---------- recogida por episodios ----------
    episodes_S, episodes_A, episodes_R, episodes_done = [], [], [], []
    episodes_trunc = []
    episodes_len = []
    pscore_list = []

    inferred_action_type: Optional[str] = None

    for ep, g in df_sorted.groupby(ep_col, sort=False):
        # Filtrado por longitud mínima
        if len(g) < min_steps:
            continue

        # Estados
        S_list = to_tensor_list(g[obs_col], dtype=float_dtype)
        S = stack_consistent(S_list, "state", dtype=float_dtype)  # (T, ..., d)
        # si viene (T,1,D) y quieres (T,D)
        if S.ndim >= 2 and S.shape[1] == 1:
            S = S.squeeze(1)

        # Acciones
        A_list = to_tensor_list(g[action_col])  # dtype puede ser float o int según el DF

        # Inferir tipo de acción si no viene dado
        if action_type is None:
            if all(t.ndim == 0 for t in A_list):
                inferred_action_type = inferred_action_type or "discrete"
            else:
                inferred_action_type = inferred_action_type or "continuous"

        # Apilar acciones
        if all(t.ndim == 0 for t in A_list):
            # Discreto (índice por paso) -> (T,)
            A = torch.stack(A_list).to(device)
            if not A.is_floating_point():  # probablemente int64 ya
                A = A.to(torch.long)
            else:
                A = A.to(torch.long)
        else:
            # Continuo -> (T, d_a)
            A = stack_consistent(A_list, "action", dtype=float_dtype)
            if A.ndim == 1:
                A = A.unsqueeze(1)

        # Recompensas, done, truncation
        R = torch.as_tensor(g[reward_col].to_list(), dtype=float_dtype, device=device)
        D_torch = torch.as_tensor(g[done_col].to_list(), dtype=torch.bool, device=device).clone()
        # MDPDataset espera dones en nd.array
        D = np.asarray(g[done_col].to_list(), dtype=bool).copy()

        # terminal: True en el último paso SIEMPRE (marca corte de trayectoria)
        Tm = torch.zeros_like(D_torch, dtype=torch.bool, device=device)
        if Tm.numel() > 0:
            Tm[-1] = True
            # si estamos truncando por longitud mínima, garantizamos fin de episodio
            D[-1] = True if truncated_col is None else D[-1]
            D_torch[-1] = True if truncated_col is None else D_torch[-1]
        if truncated_col and truncated_col in g.columns:
            # si quieres reflejar "timeout" explícito fuera
            # aquí no cambiamos 'D'; Tm ya marca fin de trayectoria
            pass

        # pscore por episodio
        P = None
        if pscore_col and pscore_col in g.columns:
            P = torch.as_tensor(g[pscore_col].to_list(), dtype=float_dtype, device=device)
        elif alt_logp_col and alt_logp_col in g.columns:
            P = torch.exp(torch.as_tensor(g[alt_logp_col].to_list(), dtype=float_dtype, device=device))
        if P is not None:
            P = torch.clamp(P, min=eps, max=1.0).to(float_dtype)
            if not torch.isfinite(P).all():
                raise ValueError(f"Episodio {ep}: pscore contiene NaN/Inf")

        episodes_S.append(S)
        episodes_A.append(A)
        episodes_R.append(R)
        episodes_done.append(D)
        episodes_trunc.append(Tm)
        episodes_len.append(len(R))
        pscore_list.append(P)

    # Dataset vacío tras el filtro
    if len(episodes_len) == 0:
        return {
            "size": 0,
            "n_trajectories": 0,
            "step_per_trajectory": None,
            "action_type": (action_type or "discrete"),
            "n_actions": None if (action_type or "discrete") != "discrete" else None,
            "action_dim": None if (action_type or "discrete") == "discrete" else None,
            "action_keys": list(action_keys) if action_keys is not None else None,
            "action_meaning": dict(action_meaning) if action_meaning is not None else None,
            "state_dim": None,
            "state_keys": list(state_keys) if state_keys is not None else None,
            "state": torch.empty(0, 0, dtype=float_dtype, device=device),
            "action": torch.empty(0, dtype=float_dtype, device=device),
            "reward": torch.empty(0, dtype=float_dtype, device=device),
            "done": np.array([], dtype=bool),
            "terminal": np.array([], dtype=bool),
            "info": None,
            "pscore": None,
            "behavior_policy": behavior_policy,
            "dataset_id": int(dataset_id),
        }

    # Tipo de acción final
    action_type_final = action_type or (inferred_action_type or "discrete")

    # step_per_trajectory
    unique_lengths = sorted(set(episodes_len))
    step_per_trajectory = int(unique_lengths[0]) if len(unique_lengths) == 1 else None

    n_trajectories = len(episodes_len)
    size = int(sum(episodes_len))

    # Aplanados
    # Estado 2D (size, d_s)
    S_flat = flatten_list_of_tensors([s if s.ndim >= 1 else s.view(-1, 1) for s in episodes_S], dtype=float_dtype)
    if S_flat.ndim == 1:
        S_flat = S_flat.view(-1, 1)
    state_dim = int(S_flat.shape[1]) if S_flat.numel() else None

    # Acción
    if action_type_final == "discrete":
        if any(a.ndim > 1 for a in episodes_A):
            raise ValueError("Se infirió 'discrete' pero hay acciones vectoriales; fija action_type='continuous'.")
        A_flat_int = flatten_list_of_tensors([a.to(torch.long) for a in episodes_A])
        if n_actions is None:
            n_actions = int(A_flat_int.max().item()) + 1 if A_flat_int.numel() else None

        if one_hot_discrete and n_actions is not None:
            A_flat = F.one_hot(A_flat_int.clamp(min=0), num_classes=n_actions).to(dtype=float_dtype)
        else:
            A_flat = A_flat_int  # (size,) long
        action_dim = None
    else:
        A_flat = flatten_list_of_tensors([a if a.ndim >= 1 else a.view(-1, 1) for a in episodes_A], dtype=float_dtype)
        if A_flat.ndim == 1:
            A_flat = A_flat.view(-1, 1)
        action_dim = int(A_flat.shape[1]) if A_flat.numel() else None
        n_actions = None

    R_flat = flatten_list_of_tensors(episodes_R, dtype=float_dtype)
    # done_flat = flatten_list_of_tensors([d.to(torch.bool) for d in episodes_done], dtype=None).to(torch.bool)
    term_flat = flatten_list_of_tensors([t.to(torch.bool) for t in episodes_trunc], dtype=None).to(torch.bool)
    # MDPDataset espera termination com nd.array, no pytorch tensor
    done_flat = flatten_list_of_arrays([d.astype(bool) for d in episodes_done]).astype(bool)
    # term_flat = flatten_list_of_arrays([t.astype(bool) for t in episodes_trunc]).astype(bool)

    # pscore
    if any(p is not None for p in pscore_list):
        P_list = []
        for T, P in zip(episodes_len, pscore_list):
            if P is None:
                P = torch.full((T,), float("nan"), dtype=float_dtype, device=device)
            P_list.append(P)
        pscore_flat = torch.cat(P_list, dim=0).to(float_dtype)
    else:
        pscore_flat = None

    # Claves opcionales (metadatos sin tensors)
    action_keys = list(action_keys) if action_keys is not None else None
    state_keys = list(state_keys) if state_keys is not None else None
    action_meaning = dict(action_meaning) if action_meaning is not None else None

    logged_dataset: Dict[str, Union[int, str, torch.Tensor, Dict]] = {
        "size": size,
        "n_trajectories": n_trajectories,
        "step_per_trajectory": step_per_trajectory,
        "action_type": action_type_final,   # "discrete" | "continuous"
        "n_actions": n_actions,             # solo para discreto
        "action_dim": action_dim,           # solo para continuo
        "action_keys": action_keys,
        "action_meaning": action_meaning,
        "state_dim": state_dim,
        "state_keys": state_keys,
        "state": S_flat,                    # torch.Tensor (size, d_s)
        "action": A_flat,                   # torch.Tensor (size,) o (size,n_actions)/(size,d_a)
        "reward": R_flat,                   # torch.Tensor (size,)
        "done": done_flat,                  # torch.Tensor (size,) bool
        "terminal": term_flat,              # torch.Tensor (size,) bool
        "info": None,
        "pscore": pscore_flat,              # torch.Tensor (size,) con NaN donde no haya
        "behavior_policy": behavior_policy,
        "dataset_id": int(dataset_id),
    }
    return logged_dataset


# ----------------------------
# 2) Adaptador RLlib -> SCOPE-RL (discreto, determinista)
# ----------------------------
# ========= Helpers =========
def _iter_states(x: np.ndarray) -> Tuple[Iterable, int]:
    arr = np.asarray(x)
    if arr.ndim == 1:
        return [arr], 1
    return arr, arr.shape[0]

def _softmax_stable(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)

# --- utils para normalizar entradas a batch ---
def _to_batch(x):
    """
    Devuelve (batch, n). Acepta dict/list/np.ndarray/estado único.
    - dict: se asume mapping de claves de obs -> arrays/listas
    - list/np.ndarray: apila a primera dimensión
    """
    if isinstance(x, dict):
        # detecta si es un único estado o batch
        first = next(iter(x.values()))
        is_single = np.asarray(first).ndim == 1
        batch = {k: (np.expand_dims(np.asarray(v), 0) if is_single else np.asarray(v))
                 for k, v in x.items()}
        n = next(iter(batch.values())).shape[0]
        return batch, n
    else:
        arr = np.asarray(x, dtype=object) if isinstance(x, list) else np.asarray(x)
        if arr.ndim == 0:
            arr = arr[None, ...]
        elif arr.ndim == 1 and arr.dtype == object:
            # lista de objetos -> apilar a lo bruto
            arr = np.stack(list(arr), axis=0)
        return arr, arr.shape[0]


# ========= Shims para d3rlpy-like API =========
class _ImplShim:
    """Backend mínimo para satisfacer expectativas de SCOPE-RL/d3rlpy."""
    def __init__(self, adapter: "RLLibPolicyAdapter"):
        self._adapter = adapter
        # d3rlpy suele usar torch.device en impl.device
        try:
            self.device = torch.device(adapter.device)
        except Exception:
            self.device = adapter.device  # mantiene el string "cpu"/"cuda" si no es torch.device

    @torch.no_grad()
    def predict_value(self, x: Union[np.ndarray, Dict]) -> np.ndarray:
        # Reenvía a la lógica del adapter
        return self._adapter.predict_value(x)
    
    @torch.no_grad()
    def predict_best_action(self, x: Union[np.ndarray, Dict]) -> np.ndarray:
        """Acción greedy por estado (discreto: índices int; continuo: vector)."""
        return self._adapter.predict_best_action(x)
    
    @torch.no_grad()
    def sample_action(self, x: Union[np.ndarray, Dict]) -> np.ndarray:
        """Opcional: acción muestreada de la policy (si la usas en pipelines estocásticos)."""
        return self._adapter.sample_action(x)


class _ConfigShim:
    """Config mínimo; d3rlpy suele consultar gamma desde config."""
    def __init__(self, gamma: float):
        self.gamma = float(gamma)


# ========= Adapter: RLlib -> interfaz mínima esperada por SCOPE-RL =========
class RLLibPolicyAdapter:
    """
    Adapter mínimo para exponer atributos/funciones que SCOPE-RL espera en `base_policy`.
    Envuelve una PPOTorchPolicy (u otra Policy discreta de RLlib).
    """

    def __init__(
        self,
        rllib_policy: Any,
        observation_shape: Any,
        action_size: int,
        gamma: float = 0.99,
        device: str = "cpu",
        action_type: str = "discrete",
    ):
        self._rllib_policy = rllib_policy
        self._observation_shape = observation_shape
        self._action_size = int(action_size)
        self._gamma = float(gamma)
        self._device = device
        self._action_type = action_type  

        # Shims que SCOPE-RL puede buscar:
        self.impl = _ImplShim(self)       # <- **clave**: añade `.impl`
        self.config = _ConfigShim(gamma)  # <- opcional pero útil

    # --- propiedades tipo d3rlpy ---
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._observation_shape

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def device(self) -> str:
        return self._device

    # --- opcional: valor de estado (útil para algunos caminos de SCOPE-RL/DM) ---
    @torch.no_grad()
    def predict_value(self, x: Union[np.ndarray, Dict]) -> np.ndarray:
        """
        Devuelve V(s) usando la value head de PPO si está disponible.
        RLlib expone 'vf_preds' en extra cuando full_fetch=True (para PPO).
        """
        # normalizamos a batch
        if isinstance(x, dict):
            first = next(iter(x.values()))
            is_single = np.asarray(first).ndim == 1
            batch = {k: (np.expand_dims(np.asarray(v), 0) if is_single else np.asarray(v)) for k, v in x.items()}
        else:
            states_iter, n = _iter_states(np.asarray(x))
            batch = np.stack(list(states_iter), axis=0) if n > 1 else np.asarray(list(states_iter)[0])[None, ...]

        # pedimos predicciones con full_fetch para obtener vf_preds
        _, _, extra = self._rllib_policy.compute_actions(batch, explore=False, full_fetch=True)
        vf = extra.get("vf_preds", None)
        if vf is None:
            # Fallback: si no está, devolvemos ceros y dejamos que FQE/DR haga su trabajo
            n = batch.shape[0] if not isinstance(batch, dict) else len(next(iter(batch.values())))
            return np.zeros((n,), dtype=np.float32)
        if isinstance(vf, torch.Tensor):
            vf = vf.cpu().numpy()
        return np.asarray(vf).reshape(-1)
    
    # --- Acción greedy (lo que pide predict_best_action del impl) ---
    @torch.no_grad()
    def predict_best_action(self, x: Union[np.ndarray, Dict]) -> np.ndarray:
        batch, n = _to_batch(x)
        acts, _, _ = self._rllib_policy.compute_actions(
            batch, explore=False, full_fetch=False
        )
        acts = np.asarray(acts)
        return acts.reshape(n, -1) if self._action_type == "continuous" and acts.ndim == 1 else acts

    # --- Muestreo estocástico (si quieres usarlo) ---
    @torch.no_grad()
    def sample_action(self, x: Union[np.ndarray, Dict]) -> np.ndarray:
        batch, n = _to_batch(x)
        acts, _, _ = self._rllib_policy.compute_actions(
            batch, explore=True, full_fetch=False
        )
        acts = np.asarray(acts)
        return acts.reshape(n, -1) if self._action_type == "continuous" and acts.ndim == 1 else acts
    
    # PPO no tiene Q explícito
    def predict_q(self, x: Union[np.ndarray, Dict]) -> Optional[np.ndarray]:
        return None


@dataclass
class RLlibCategoricalHead(BaseHead):
    """
    Head estocástico para políticas discretas de RLlib (p.ej. PPOTorchPolicy).
    Expone pi(a|s), muestreo, greedy, y pscore coherentes con SCOPE-RL.
    """
    name: str = "rllib_categorical"
    rllib_policy: Any = None          # algo.get_policy()
    base_policy: Any = None   
    action_size: int = None
    observation_shape: Any = None           
    action_type: str = "discrete"
    temperature: float = 1.0          # opcional (T=1 por defecto)

    def __post_init__(self):
        # inferir action_size si falta
        if self.action_size is None:
            try:
                self.action_size = int(self.rllib_policy.action_space.n)
            except Exception:
                raise ValueError("action_size no especificado y no se pudo inferir de rllib_policy.action_space.n")

        # inferir observation_shape si falta (de la policy o pásalo tú)
        if self.observation_shape is None:
            try:
                self.observation_shape = tuple(self.rllib_policy.observation_space.shape)
            except Exception:
                raise ValueError("observation_shape no especificado y no se pudo inferir de rllib_policy.observation_space.shape")

        # construir base_policy como adapter (en vez de usar la policy de RLlib directa)
        if self.base_policy is None:
            self.base_policy = RLLibPolicyAdapter(
                rllib_policy=self.rllib_policy,
                observation_shape=self.observation_shape,
                action_size=self.action_size,
                gamma=0.9,
                device="cpu",  # cambia a "cuda" si procede
            )

    # --- evitar deepcopy de la policy ---
    def __deepcopy__(self, memo):
        new = RLlibCategoricalHead(
            name=self.name,
            rllib_policy=self.rllib_policy,    # misma ref
            action_size=self.action_size,
            temperature=self.temperature,
        )
        memo[id(self)] = new
        return new

    # opcional: pickle-safe
    def __getstate__(self):
        d = self.__dict__.copy()
        d["rllib_policy"] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)

    # -------- internals ----------
    @torch.no_grad()
    def _batch_logits(self, x: np.ndarray) -> np.ndarray:
        states_iter, n = _iter_states(x)
        # RLlib acepta lista/np para batch
        batch = np.stack(list(states_iter), axis=0) if n > 1 else np.asarray(list(states_iter)[0])[None, ...]
        # full_fetch=True para obtener 'action_dist_inputs' (logits)
        _, _, extra = self.rllib_policy.compute_actions(
            batch, explore=False, full_fetch=True
        )
        logits = extra.get("action_dist_inputs", None)
        if logits is None:
            # Algunos policies devuelven tensor torch; convertimos
            action_dist = extra.get("action_dist", None)
            if action_dist is not None and hasattr(action_dist, "inputs"):
                logits = action_dist.inputs
            elif action_dist is not None and hasattr(action_dist, "logits"):
                logits = action_dist.logits
            if logits is None:
                raise RuntimeError("No se pudieron obtener logits ('action_dist_inputs'). Usa full_fetch=True y una policy discreta.")
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        else:
            logits = np.asarray(logits)
        if self.temperature != 1.0:
            logits = logits / float(self.temperature)
        # sanity
        if logits.shape[-1] != int(self.action_size):
            raise RuntimeError(f"Dimensión de acciones inesperada: {logits.shape[-1]} != {self.action_size}")
        return logits

    def _batch_probs(self, x: np.ndarray) -> np.ndarray:
        return _softmax_stable(self._batch_logits(x))

    # -------- Métodos requeridos por BaseHead (discrete) --------

    def calc_action_choice_probability(self, x: np.ndarray) -> np.ndarray:
        """
        Devuelve matriz (n, A) con pi(a|s) para cada estado.
        (SCOPE-RL usa esto para IS/WIS/DR.)
        """
        return self._batch_probs(x)

    def calc_pscore_given_action(self, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Devuelve pi(a_t|s_t) para cada par (s_t, a_t) del dataset.
        """
        probs = self._batch_probs(x)
        a = np.asarray(action).astype(int).reshape(-1)
        return np.clip(probs[np.arange(len(a)), a], 1e-12, 1.0)

    def predict_online(self, x: np.ndarray) -> np.ndarray:
        """
        Devuelve acción greedy (argmax pi) por estado. Shape: (n,)
        """
        probs = self._batch_probs(x)
        return probs.argmax(axis=-1).astype(int)

    def sample_action_and_output_pscore(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devuelve (acciones muestreadas, pscore) donde pscore = pi(a_sampled|s).
        """
        probs = self._batch_probs(x)
        n, A = probs.shape
        # muestreo categórico vectorizado
        # (truco: compara cumsum con uniformes)
        u = np.random.rand(n, 1)
        cdf = np.cumsum(probs, axis=1)
        acts = (u > cdf).sum(axis=1).astype(int)
        pscore = np.clip(probs[np.arange(n), acts], 1e-12, 1.0)
        return acts, pscore

    # --- Método que te falta y que SCOPE-RL a veces llama explícitamente ---
    def sample_action(self, x_single: Union[np.ndarray, Dict]) -> int:
        """
        Acción muestreada para un único estado (SCOPE-RL lo puede invocar).
        """
        return self.predict_online(x_single)

# ----------------------------
# 3) Carga policy RLlib 2.11 + dataset desde JSON -> OPE
# ----------------------------
BEH_EPISODES_JSON = "/opt/ml/code/episodes/120820251600/140825_generated_rllib_ppo_rllib_seed_0000_1000eps_200steps_exp_0/output-2025-08-14_12-19-12_worker-0_0.json"  
CKPT_DIR_EVAL        = "/opt/ml/code/checkpoints/130820251600"          # eval policy
CKPT_DIR_BEH        = "/opt/ml/code/checkpoints/120820251600"          # behavioral policy
ENV_ID          = "LunarLander-v3"                    # <-- usa tu env real

# Carga policy RLlib
algo = Algorithm.from_checkpoint(CKPT_DIR_EVAL)
rllib_policy_eval = algo.get_policy()
print("policy.observation_space:", algo.get_policy().observation_space)

algo = Algorithm.from_checkpoint(CKPT_DIR_BEH)
rllib_policy_beh = algo.get_policy()


# Entorno (se usa para dimensiones/espacios; no generamos datos del env)
env = gym.make(ENV_ID)
print("env.observation_space.shape:", env.observation_space.shape)
action_size = env.action_space.n  # discreto

# Carga logged_dataset desde el JSON de RLlib
# logged_dataset = load_rllib_logged_dataset(RLlib_JSON_PATH)

# Carga logged_dataset desde Pandas DF
reader_beh = JsonReader(BEH_EPISODES_JSON)
beh_df = load_json_to_df(reader_beh, 1000)
logged_dataset = df_to_logged_dataset(beh_df)
# logged_dataset = df_to_logged_dataset_torch(beh_df)

# Inyecta la policy como evaluation_policy
# beh_head = RLlibCategoricalHead(rllib_policy=rllib_policy_beh, action_size=action_size)
eval_head = RLlibCategoricalHead(rllib_policy=rllib_policy_eval,
                                action_size=action_size,
                                observation_shape=env.observation_space.shape)


# generamos datos de la politica beh

# Prepara inputs para OPE
prep = CreateOPEInput(env=env)
input_dict = prep.obtain_whole_inputs(
    logged_dataset=logged_dataset,
    evaluation_policies=[eval_head],
    require_value_prediction=False,  # activa si vas a usar DM/DR (entrena FQE)
    n_trajectories_on_policy_evaluation=100,  # si no necesitas rollouts on-policy sintéticos
    random_state=123,
)

# Ejecuta OPE (estimadores discretos de ejemplo)
# estimators = [DM(), SNPDIS(), SNDR()]
estimators = [SNPDIS()]
ope = OffPolicyEvaluation(logged_dataset=logged_dataset, ope_estimators=estimators)

policy_value_df, policy_value_interval_df = ope.summarize_off_policy_estimates(
    input_dict=input_dict,
    random_state=123,
)

print(policy_value_df[eval_head.name])
print(policy_value_interval_df[eval_head.name])



