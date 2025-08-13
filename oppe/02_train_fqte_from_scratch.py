import numpy as np, os
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import debugpy
import pandas as pd

import ray

from statistics import mean
from oppe_utils import load_checkpoint, load_json_to_df, calculate_value_function
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.algorithms import Algorithm

# Aseguramos la reproducibilidad (opcional)
torch.manual_seed(0)
np.random.seed(0)

# Supongamos que ya tenemos un DataFrame `df` con las columnas:
# 'state', 'action', 'reward', 'next_state', 'done'.
# Además, supongamos que tenemos acceso a la política objetivo PPO a través de una función:
# policy_action(state) -> que devuelve la acción (entera) que la política PPO tomaría en ese estado.
#
# NOTA: En este script, 'state' y 'next_state' pueden ser vectores NumPy o listas de números (observaciones).
#       Se asume espacio de acciones discreto (acciones representadas por enteros 0,1,...,N-1).


# == Definición del modelo FQE (red neuronal Q) ==
class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_size=64):
        super(QNetwork, self).__init__()
        # Red feed-forward simple: dos capas ocultas y salida de tamaño num_actions
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
    def forward(self, state):
        # Pase hacia adelante. Si el estado es un vector 1D, expandir a 2D (batch de tamaño 1).
        return self.net(state)


def train_nn(df, policy_action, save_dir, resume_training=1):

    # == Preparación de los datos ==
    # Convertimos las columnas del DataFrame a tensores de PyTorch para entrenamiento.
    states = torch.tensor(np.stack(df['obs'].values), dtype=torch.float32)
    actions = torch.tensor(df['action'].values, dtype=torch.int64)   # acciones como enteros (índices)
    rewards = torch.tensor(df['reward'].values, dtype=torch.float32)
    next_states = torch.tensor(np.stack(df['next_state'].values), dtype=torch.float32)
    dones = torch.tensor(df['done'].values, dtype=torch.float32)     # 1.0 si es terminal, 0.0 si no


    # Determinamos la dimensionalidad de estado y número de acciones.
    state_dim = states.shape[2] if states.ndim > 1 else 1
    num_actions = int(actions.max().item() + 1)  # suponiendo acciones indexadas desde 0 hasta max.

    print(f"Dimensión del estado: {state_dim}, Número de acciones: {num_actions}")
    
    # Inicializamos la red Q y una red "target" para estabilizar el entrenamiento.
    q_net = QNetwork(state_dim, num_actions)
    target_net = QNetwork(state_dim, num_actions)
    target_net.load_state_dict(q_net.state_dict())  # inicializar target con mismos pesos
    target_net.eval()  # la red target no se entrena (se actualiza periódicamente)

    # Definimos el optimizador y los hiperparámetros.
    optimizer = optim.Adam(q_net.parameters(), lr=0.0005)  # tasa de aprendizaje elegida (por ejemplo 1e-3)
    criterion = nn.MSELoss()  # utilizaremos MSE para el error temporal diferencial
    gamma = 0.99              # factor de descuento (ajustar según el entorno)

    # Opciones de entrenamiento
    batch_size = 512
    num_epochs = 100  # número de épocas de entrenamiento (pasadas completas por el dataset)
    target_update_interval = 10  # cada cuántas épocas sincronizar la red target con la principal
    save_every = 20

    # Leyendo estado anterior si existe 
    if os.path.exists(save_dir) and resume_training==1:
        checkpoint = torch.load(save_dir + '/fqe_epoch_50.pt')
        q_net.load_state_dict(checkpoint["model_state_dict"])
        target_net.load_state_dict(checkpoint["target_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Se cargó el checkpoint desde {save_dir}, reanudando en la época {start_epoch}")
    else:
        start_epoch = 0
        print("No se encontró checkpoint, entrenamiento desde cero.")

    # == Entrenamiento FQE ==
    for epoch in range(start_epoch, num_epochs):
        # Barajamos los índices de los datos para mezclar los episodios en cada época
        indices = torch.randperm(states.shape[0])
        batch_losses = []
        # Procesamos en mini-batches
        for i in range(0, states.shape[0], batch_size):
            batch_idx = indices[i : i+batch_size]
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_rewards = rewards[batch_idx]
            batch_next_states = next_states[batch_idx]
            batch_dones = dones[batch_idx]

            # Computamos Q(s,a) actual (predicción de la red) para las acciones ejecutadas en el batch
            # q_values tiene dimensión [batch_size, num_actions], tomamos la columna correspondiente a batch_actions
            q_values = q_net(batch_states).squeeze(1)                        # shape: (batch, num_actions)
            q_sa = q_values.gather(1, batch_actions.view(-1,1)).squeeze(1)  # Q(s,a) predicho

            # Computamos el target de Q(s,a) usando la actualización de Bellman con la política PPO.
            # Para cada transición, si done=1 (estado terminal), target = reward (no hay futuro).
            # Si no es terminal, target = reward + gamma * Q_target(s', a'), donde a' = acción recomendada por la política PPO en s'.
            with torch.no_grad():
                # Obtenemos las acciones a' que la política PPO tomaría en los next_states
                # (Aquí asumimos que policy_action puede procesar un estado a la vez; si es vectorial se puede vectorizar este paso)
                next_actions = []
                for ns in batch_next_states:
                    # Convertimos ns (tensor) a formato apropiado para la política (por ejemplo a numpy array)
                    a_prime,_,info = policy_action.compute_single_action(ns, explore=False)       # acción según la política PPO
                    next_actions.append(a_prime)
                next_actions = torch.tensor(next_actions, dtype=torch.int64)

                # Calculamos Q_target(s', a') para cada transición del batch
                q_next = target_net(batch_next_states)                   # Q_target(s', :) para cada estado del batch
                q_next_sa = q_next.gather(1, next_actions.view(-1,1)).squeeze(1)  # Q_target(s', a' recomendado)
                # Si el estado s' es terminal (done=1), no hay contribución futura: multiplicamos por (1-done) para anular.
                q_next_sa = q_next_sa * (1 - batch_dones)
                # Definimos el valor objetivo (target) de Q(s,a)
                target_values = batch_rewards + gamma * q_next_sa

            # Calculamos la pérdida (error cuadrático) entre Q(s,a) predicho y el target calculado
            loss = criterion(q_sa, target_values)
            batch_losses.append(loss.item())

            # Retropropagación y optimización de la red Q
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Fin del loop de mini-batches

        # Actualización periódica de la red target para seguir a la red principal
        if (epoch + 1) % target_update_interval == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (epoch + 1) % save_every == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": q_net.state_dict(),
                "target_model_state_dict": target_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": np.mean(batch_losses)
                # (opcional) añade otros elementos que quieras guardar, como avg_loss
            }
            checkpoint_path = os.path.join(save_dir, f"fqe_epoch_{epoch+1}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint guardado en {checkpoint_path}")

        # (Opcional) Mostrar la pérdida media de la época para monitorear convergencia
        if (epoch + 1) % target_update_interval == 0 or epoch == 0:
            avg_loss = np.mean(batch_losses)
            print(f"Época {epoch+1}/{num_epochs} - Pérdida promedio: {avg_loss:.4f}")
    return q_net



def evaluate_policy(q_net, df, policy_action):

    states = torch.tensor(np.stack(df['obs'].values), dtype=torch.float32)

    # == Evaluación de la política PPO usando el modelo Q entrenado ==
    q_net.eval()  # ponemos la red en modo evaluación
    # Identificamos estados iniciales de cada episodio en el DataFrame.
    # Suponemos que el DataFrame está ordenado por episodios, y que 'done' indica fin de episodio.
    initial_state_indices = []
    N = len(df)
    for i in range(N):
        if i == 0 or df.iloc[i-1]['done']:  # es el primer paso de un episodio si es el inicio del df o el registro anterior tuvo done=True
            initial_state_indices.append(i)

    initial_states = states[initial_state_indices]
    # Calculamos el valor promedio de la política PPO desde los estados iniciales.
    values = []
    for s in initial_states:
        # Obtenemos la acción sugerida por la política PPO en el estado inicial
        a,_,info = policy_action.compute_single_action(s, explore=False)  # acción (entera) de la política PPO
        # Calculamos Q(s, a) con nuestra red entrenada
        q_value = q_net(s)  # hacemos forward con tamaño de batch 1
        q_value_sa = q_value[0, a].item()
        values.append(q_value_sa)

    if values:
        estimated_value = np.mean(values)
        print(f"\nValor esperado estimado de la política PPO (retorno promedio inicial): {estimated_value:.3f}")
    else:
        print("No se encontraron estados iniciales para evaluar la política.")


def oppe():

    BEH_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/310720251600"
    EVAL_CHECKPOINT_PATH = "/opt/ml/code/checkpoints/300720251000"
    
    BEH_EPISODES_JSON_TRAIN = '/opt/ml/code/episodes/310720251600/310725_generated_rllib_ppo_rllib_seed_0000_2000eps_200steps_exp_0'
    BEH_EPISODES_JSON_TEST = '/opt/ml/code/episodes/310720251600/310725_generated_rllib_ppo_rllib_seed_0000_200eps_200steps_exp_0'
    BEH_EPISODES_JSON = '/opt/ml/code/episodes/310720251600/010825_generated_rllib_ppo_rllib_seed_0000_1000eps_200steps_exp_0'
    EVAL_EPISODES_JSON = '/opt/ml/code/episodes/300720251000/050825_generated_rllib_ppo_rllib_seed_0000_1000eps_200steps_exp_0'
    

    # beh_agent = load_checkpoint(BEH_CHECKPOINT_PATH)
    eval_agent = load_checkpoint(EVAL_CHECKPOINT_PATH)
  
    # ---------------------------------------------------------------------------#
    # 1. Real Expected Value for the policies based in the current json episodes 
    # --------------------------------------------------------------------------- #
    reader_beh = JsonReader(BEH_EPISODES_JSON)
    reader_beh_train = JsonReader(BEH_EPISODES_JSON_TRAIN)
    reader_beh_test = JsonReader(BEH_EPISODES_JSON_TEST)
    reader_target = JsonReader(EVAL_EPISODES_JSON)
    beh_eps_df = load_json_to_df(reader_beh, 1000)
    target_eps_df = load_json_to_df(reader_target, 2)
    beh_expected_return = calculate_value_function(beh_eps_df, 0.99)
    target_expected_return = calculate_value_function(target_eps_df, 0.99)
    print(f"Avg_Expecting_Return (BEH_POLICY) Value - RLLIB Generated episodes: {beh_expected_return}")
    print(f"Avg_Expecting_Return (TARGET_POLICY) Value - RLLIB Generated episodes: {target_expected_return}")

    # --------------------------------------------------------------------------- #
    # 2. ENTRENAR ESTIMADOR DM (requiere Q‑model)
    # --------------------------------------------------------------------------- #
   
    print("\n⏳ Entrenando Q‑model (DM)...")
    SAVE_DIR = "./fqe_checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    USE_TRAINED_FQTE = 0
    if USE_TRAINED_FQTE == 0:
        beh_train_df = load_json_to_df(reader_beh_train, 2000)
        q_net = train_nn(beh_train_df, eval_agent, SAVE_DIR)
    else:
        if os.path.exists(SAVE_DIR):
            checkpoint = torch.load(SAVE_DIR + '/fqe_epoch_15.pt')
            q_net = QNetwork(8, 4)
            q_net.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"]
            avg_loss = checkpoint["avg_loss"]
            print(f"Se cargó el checkpoint desde {SAVE_DIR}, entrenadas {start_epoch} épocas con Avg. Loss {avg_loss}")
    
    beh_test_df = load_json_to_df(reader_beh_test, 200)
    evaluate_policy(q_net, beh_test_df, eval_agent)
 
    
if __name__ == '__main__':
    
    debug = 0

    if debug == 1:
        # Escucha en el puerto 5678 (puedes cambiarlo)
        debugpy.listen(("0.0.0.0", 5678))
        print("Esperando debugger de VS Code para conectar...")
        debugpy.wait_for_client()
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    print(ray.__version__)

    oppe()
