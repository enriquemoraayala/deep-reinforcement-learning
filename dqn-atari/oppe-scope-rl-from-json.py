# pip install "ray[rllib]==2.11.0" scope-rl d3rlpy gymnasium numpy

import json
import numpy as np
import pandas as pd
import debugpy
from typing import Any, Tuple, Iterable
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
    pscore_col: Optional[str] = "action_prob",  # prob(a_t | s_t) del behavior policy (si la tienes)
) -> List[Dict[str, np.ndarray]]:
    """
    Convierte un DF con una fila por (episodio, paso) en una lista de episodios:
      {'state','action','reward','next_state','done',('pscore')}
    - Asume que obs/next_state pueden ser arrays/listas; se hace stack por episodio.
    - Si next_state tiene nulos o no cuadra, se deriva por desplazamiento de state.
    """
    assert {ep_col, step_col, obs_col, action_col, reward_col, done_col}.issubset(df.columns), \
        "Faltan columnas obligatorias en el DataFrame."

    episodes = []
    for ep, g in df.sort_values([ep_col, step_col]).groupby(ep_col):
        # Helper para apilar en np.array con conversión robusta
        def to_array(series):
            vals = series.to_list()
            # Si los elementos ya son arrays/listas/escalares, normalizamos:
            vals = [np.asarray(v) if not isinstance(v, (np.ndarray,)) else v for v in vals]
            # Si son escalares -> array 1D; si son vectores -> stack
            try:
                return np.stack(vals)
            except Exception:
                # Si hay mezcla de formas, forzamos np.array con dtype=object (último recurso)
                return np.array(vals, dtype=object)

        S = to_array(g[obs_col])
        A = to_array(g[action_col])
        R = np.asarray(g[reward_col].to_list(), dtype=float)
        D = np.asarray(g[done_col].to_list(), dtype=bool)

        # next_state: usar el del DF si está completo y sin nulos; si no, derivarlo
        NS_available = next_obs_col in g and g[next_obs_col].notna().all()
        if NS_available:
            NS = to_array(g[next_obs_col])
            shapes_ok = (hasattr(NS, "shape") and hasattr(S, "shape") and len(NS) == len(S))
        else:
            shapes_ok = False

        if not shapes_ok:
            # Derivar por desplazamiento: NS[t] = S[t+1], y el último se duplica
            if isinstance(S, np.ndarray) and len(S) >= 1 and S.dtype != object:
                NS = np.vstack([S[1:], S[-1:]])
            else:
                # Versión robusta también para dtype=object
                NS = np.array(list(S[1:]) + [S[-1]], dtype=object)

        ep_dict = {"state": S, "action": A, "reward": R, "next_state": NS, "done": D}

        if pscore_col and pscore_col in g.columns:
            P = np.asarray(g[pscore_col].to_list())
            # Si P viene como prob escalar de la acción tomada, está perfecto para IS/DR.
            ep_dict["pscore"] = P

        episodes.append(ep_dict)

    return episodes

# ----------------------------
# 2) Adaptador RLlib -> SCOPE-RL (discreto, determinista)
# ----------------------------
# helper para asegurar que Scope-rl envia algo iterable a la clase
def _iter_states(x) -> Tuple[Iterable[Any], int]:
    """
    Normaliza x en un iterable de estados y devuelve también n.
    Soporta:
      - dict con 'state'/'obs'/'observation'
      - list/tuple de estados
      - np.ndarray (cualquier dtype, incluido object)
      - un único estado (se envuelve como lista de 1)
    """
    # dict -> intenta extraer el array/lista de estados
    if isinstance(x, dict):
        for k in ("state", "obs", "observation"):
            if k in x:
                states = x[k]
                break
        else:
            # no hay clave conocida: toma el primer valor
            states = next(iter(x.values()))
    else:
        states = x

    # convertir a iterable y obtener n
    if isinstance(states, (list, tuple)):
        return states, len(states)

    if isinstance(states, np.ndarray):
        if states.ndim == 0:
            # escalar -> un solo estado
            return [states.item()], 1
        if states.ndim == 1 and states.dtype == object:
            # vector de objetos (cada elemento puede ser un dict o array)
            return list(states), len(states)
        # matriz (n, ...) -> iterar filas
        return (states[i] for i in range(states.shape[0])), states.shape[0]

    # caso “un solo estado” (dict/array/lo que sea)
    return [states], 1



@dataclass
class RLlibGreedyHead(BaseHead):
    """Adaptador mínimo para usar una policy de RLlib como evaluation_policy en SCOPE-RL (acción discreta)."""
    name: str = "rllib_greedy"
    rllib_policy: Any = None         # instancia de RLlib Policy (p. ej. algo.get_policy())
    action_size: int = None          # n° de acciones discretas


    # --- parche clave: evitar deepcopy de la policy ---
    def __deepcopy__(self, memo):
        # Creamos una nueva instancia copiando solo campos “seguros”,
        # y reusamos la MISMA referencia a rllib_policy.
        new = RLlibGreedyHead(
            name=self.name,
            rllib_policy=self.rllib_policy,   # <- misma referencia, no deepcopy
            action_size=self.action_size,
        )
        memo[id(self)] = new
        return new

    # --- opcional: hacer que el objeto sea “pickle-safe” si SCOPE-RL paraleliza ---
    def __getstate__(self):
        d = self.__dict__.copy()
        # No intentes picklear la policy de RLlib (no es necesaria para serializar el “head”)
        d["rllib_policy"] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        # El llamador debe volver a asignar la policy si hace falta tras un pickle/unpickle.
        # En la práctica, SCOPE-RL no debería necesitar picklear el head con la policy dentro.

    # --- Obligatorios para BaseHead en flujos discretos ---

    def calc_action_choice_probability(self, x: np.ndarray) -> np.ndarray:
        """Devuelve prob(a|s) con una distribución one-hot de la acción greedy (determinista)."""
        states_iter, n = _iter_states(x)
        probs = np.zeros((n, self.action_size), dtype=float)
        i = 0
        for s in states_iter:
            a, _, _ = self.rllib_policy.compute_single_action(s, explore=False)
            probs[i, int(a)] = 1.0
            i += 1
        return probs

    def calc_pscore_given_action(self, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Devuelve pscore de la acción observada bajo la policy (1 si coincide con la greedy, 0 en otro caso)."""
        states_iter, n = _iter_states(x)
        action = np.asarray(action).reshape(-1)
        p = np.zeros((n,), dtype=float)
        i = 0
        for s in states_iter:
            a_star, _, _ = self.rllib_policy.compute_single_action(s, explore=False)
            p[i] = 1.0 if int(a_star) == int(action[i]) else 0.0
            i += 1
        return p

    def predict_online(self, x: np.ndarray) -> np.ndarray:
        """Acción greedy por estado (vector de shape (n,))."""
        states_iter, n = _iter_states(x)
        acts = np.zeros((n,), dtype=int)
        i = 0
        for s in states_iter:
            a, _, _ = self.rllib_policy.compute_single_action(s, explore=False)
            acts[i] = int(a)
            i += 1
        return acts
    def sample_action_and_output_pscore(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Requerido por BaseHead.
        Devuelve (acciones, pscore_de_acciones) para los estados x, donde pscore es prob(a|s) bajo esta policy.
        Como es determinista (greedy), pscore = 1.0 para la acción devuelta.
        """
        states_iter, n = _iter_states(x)
        acts = np.zeros((n,), dtype=int)
        pscore = np.ones((n,), dtype=float)  # determinista
        i = 0
        for s in states_iter:
            a, _, _ = self.rllib_policy.compute_single_action(s, explore=False)
            acts[i] = int(a)
            i += 1
        return acts, pscore

# ----------------------------
# 3) Carga policy RLlib 2.11 + dataset desde JSON -> OPE
# ----------------------------
BEH_EPISODES_JSON = "/opt/ml/code/episodes/120820251600/140825_generated_rllib_ppo_rllib_seed_0000_1000eps_200steps_exp_0/output-2025-08-14_12-19-12_worker-0_0.json"  
CKPT_DIR        = "/opt/ml/code/checkpoints/130820251600"          # behavioral policy
ENV_ID          = "LunarLander-v3"                    # <-- usa tu env real

# Carga policy RLlib
algo = Algorithm.from_checkpoint(CKPT_DIR)
rllib_policy = algo.get_policy()

# Entorno (se usa para dimensiones/espacios; no generamos datos del env)
env = gym.make(ENV_ID)
action_size = env.action_space.n  # discreto

# Carga logged_dataset desde el JSON de RLlib
# logged_dataset = load_rllib_logged_dataset(RLlib_JSON_PATH)

# Carga logged_dataset desde Pandas DF
reader_beh = JsonReader(BEH_EPISODES_JSON)
beh_df = load_json_to_df(reader_beh, 1000)
logged_dataset = df_to_logged_dataset(beh_df)

# Inyecta la policy como evaluation_policy
eval_head = RLlibGreedyHead(rllib_policy=rllib_policy, action_size=action_size)

# Prepara inputs para OPE
prep = CreateOPEInput(env=env)
input_dict = prep.obtain_whole_inputs(
    logged_dataset=logged_dataset,
    evaluation_policies=[eval_head],
    # require_value_prediction=True,  # activa si vas a usar DM/DR (entrena FQE)
    n_trajectories_on_policy_evaluation=0,  # si no necesitas rollouts on-policy sintéticos
    random_state=123,
)

# Ejecuta OPE (estimadores discretos de ejemplo)
estimators = [DM(), SNPDIS(), SNDR()]
ope = OffPolicyEvaluation(logged_dataset=logged_dataset, ope_estimators=estimators)

policy_value_df, policy_value_interval_df = ope.summarize_off_policy_estimates(
    input_dict=input_dict,
    random_state=123,
)

print(policy_value_df[eval_head.name])
print(policy_value_interval_df[eval_head.name])



