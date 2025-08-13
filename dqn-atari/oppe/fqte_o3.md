Perfecto. Voy a prepararte un script en Python que implemente el método Direct Method con Fitted Q Evaluation (FQE) usando PyTorch, entrenando el modelo FQE a partir de un DataFrame con episodios y evaluando la política PPO objetivo. Incluiré también una explicación teórica detallada sobre FQE y el método Direct Method.

Te aviso en cuanto lo tenga listo.


# Evaluación de Políticas con **Método Directo** y **Fitted Q Evaluation (FQE)** en PyTorch

## Explicación Teórica

El **método directo (Direct Method, DM)** es la forma más sencilla de realizar **evaluación de políticas fuera de línea** (*off-policy evaluation*, OPE). Consiste en modelar explícitamente algún aspecto del comportamiento de la política objetivo (por ejemplo, modelar el entorno o la función $Q$) y entrenar ese modelo con datos históricos, para luego estimar directamente el valor de la política objetivo mediante dicho modelo. En enfoques de DM **basados en la función $Q$** (modelo *free*), se aprende directamente la función de valor **estado-acción** $Q^\pi(s,a)$ de la política $\pi$ que deseamos evaluar. Una vez aprendida $\hat{Q}^\pi$, se calcula el desempeño estimado de la política *plug-in* en base a la definición $ \eta^{\pi} = \mathbb{E}\_{s \sim \mathbb{G},, a \sim \pi(\cdot|s)} Q^\pi(s,a)$ (valor esperado de $Q$ cuando $s$ se distribuye según el estado inicial y $a$ se elige según la política).

Un método destacado en esta categoría es **Fitted Q Evaluation (FQE)**. FQE es un algoritmo de evaluación off-policy que aprende iterativamente la función $Q^\pi$ resolviendo la **ecuación de Bellman de la política objetivo**. Recordemos que $Q^\pi(s,a)$ se define como el valor esperado (descontado) de recompensas futuras al comenzar en estado $s$, tomar acción $a$ y seguir la política $\pi$ posteriormente. Formalmente:

$Q^\pi(s,a) = \mathbb{E}^{\pi}\Big[\sum_{t=0}^{\infty} \gamma^t R_t \,\Big|\, S_0=s, A_0=a\Big]\,.$

Esta función satisface la **ecuación de Bellman bajo $\pi$**:

$Q^\pi(s,a) = \mathbb{E}^{\pi}\big[R_t + \gamma\,Q^\pi(S_{t+1}, A_{t+1}) \mid S_t=s, A_t=a\big]\,,$

donde $A\_{t+1}\sim \pi(\cdot \mid S\_{t+1})$. En palabras, el valor $Q^\pi(s,a)$ debe igualar la recompensa inmediata $R\_t$ más el valor descontado $\gamma$ del siguiente estado $S\_{t+1}$, suponiendo que el agente seguirá la política $\pi$ a futuro.

**Fitted Q Evaluation** aprovecha que $Q^\pi$ es la única solución de punto fijo de esa ecuación de Bellman. El algoritmo comienza con una estimación inicial $\widehat{Q}^0$ (por ejemplo, inicializar la red neuronal $Q$ con pesos aleatorios) y luego repite iterativamente un ajuste de regresión para acercarse al punto fijo. En cada iteración $\ell$, se **minimiza el error cuadrático de Bellman** sobre los datos offline: cada transición $(s,a,r,s',\text{done})$ contribuye un término

$\Big(r + \gamma \, \mathbb{E}_{a' \sim \pi(\cdot|s')} \widehat{Q}^{\,\ell-1}(s',a') \;-\; Q(s,a)\Big)^2,$

que se suma en el loss a minimizar. En la práctica, esto significa que ajustamos $\widehat{Q}$ para predecir **el retorno observado $r$ más el valor futuro estimado** desde el siguiente estado $s'$ (si el episodio no terminó) siguiendo la **política objetivo** $\pi$. Este proceso iterativo continúa hasta la convergencia, obteniendo una aproximación $\widehat{Q}^{\pi}\_{FQE}$ del verdadero $Q^\pi$. FQE suele dar buenos resultados si la clase de función $Q$ (por ejemplo, la arquitectura de la red neuronal) es suficientemente expresiva, aunque puede introducir cierto sesgo si el modelo $Q$ no se ajusta bien a la dinámica del entorno.

**Integración de FQE con la política PPO:** En nuestro caso, la política objetivo $\pi$ es una política entrenada mediante PPO (Proximal Policy Optimization). Para evaluar $\pi$ con FQE, debemos utilizar dicha política en el cálculo de los valores esperados futuros. Concretamente, durante el entrenamiento de FQE, para cada transición usamos la política PPO para determinar la acción $a' = \pi(s')$ que la política tomaría en el siguiente estado $s'$. De este modo, el **target de entrenamiento** para $Q(s,a)$ será $y = r + \gamma , Q\_{\text{target}}(s', a')$ (si el estado $s'$ no es terminal), donde $Q\_{\text{target}}$ es la estimación actual (o de la iteración previa) de la función $Q$. Aquí es fundamental que $a'$ provenga de la **política objetivo PPO**, garantizando que $\widehat{Q}$ aprenda a valorar estados y acciones **bajo la suposición de que a futuro se seguirá la política PPO**. Si la política PPO es estocástica y podemos obtener su distribución de probabilidades $\pi(a|s')$, lo ideal sería calcular la expectativa $\mathbb{E}\_{a' \sim \pi(\cdot|s')}Q(s',a')$ sumando las $Q$-valores ponderados por dichas probabilidades. En caso de no tener directamente las probabilidades, una aproximación común es tomar la acción más probable o muestrear una acción $a'$ de $\pi(s')$ para usar en el cálculo del target.

Una vez entrenada la red neural que aproxima $Q^\pi$, podemos estimar el **valor esperado de la política PPO**. Por ejemplo, si conocemos el estado inicial $s\_0$ (o un conjunto de estados iniciales), calculamos $V^\pi(s\_0) = \mathbb{E}*{a \sim \pi(\cdot|s\_0)} \hat{Q}^\pi(s\_0, a)$. En la práctica, esto puede hacerse muestreando acciones iniciales de la política o utilizando la acción determinística de PPO en $s\_0$. Si tenemos múltiples episodios de prueba, podemos estimar $\eta^\pi$ promediando $\hat{Q}^\pi(s*{0}^{(i)}, a\_{0}^{(i)})$ para los estados iniciales $s\_{0}^{(i)}$ de cada episodio, con $a\_{0}^{(i)}$ tomada según la política PPO. Esta integración de FQE con PPO nos brinda una estimación directa del retorno esperado de la política PPO sin necesidad de desplegarla en el entorno real.

A continuación, proporcionamos la **implementación en código Python** del método directo con FQE usando PyTorch. El código incluye: la definición de la red neuronal $Q$, el bucle de entrenamiento que ajusta la red mediante FQE usando los datos offline (DataFrame con `state`, `action`, `reward`, `next_state`, `done`), y finalmente la evaluación de la política PPO usando el modelo entrenado. Los comentarios explican cada parte del proceso.

## Implementación en PyTorch

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

# == Preparación de los datos ==
# Convertimos las columnas del DataFrame a tensores de PyTorch para entrenamiento.
states = torch.tensor(np.stack(df['state'].values), dtype=torch.float32)
actions = torch.tensor(df['action'].values, dtype=torch.int64)   # acciones como enteros (índices)
rewards = torch.tensor(df['reward'].values, dtype=torch.float32)
next_states = torch.tensor(np.stack(df['next_state'].values), dtype=torch.float32)
dones = torch.tensor(df['done'].values, dtype=torch.float32)     # 1.0 si es terminal, 0.0 si no

# Determinamos la dimensionalidad de estado y número de acciones.
state_dim = states.shape[1] if states.ndim > 1 else 1
num_actions = int(actions.max().item() + 1)  # suponiendo acciones indexadas desde 0 hasta max.

print(f"Dimensión del estado: {state_dim}, Número de acciones: {num_actions}")

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

# Inicializamos la red Q y una red "target" para estabilizar el entrenamiento.
q_net = QNetwork(state_dim, num_actions)
target_net = QNetwork(state_dim, num_actions)
target_net.load_state_dict(q_net.state_dict())  # inicializar target con mismos pesos
target_net.eval()  # la red target no se entrena (se actualiza periódicamente)

# Definimos el optimizador y los hiperparámetros.
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)   # tasa de aprendizaje elegida (por ejemplo 1e-3)
criterion = nn.MSELoss()  # utilizaremos MSE para el error temporal diferencial
gamma = 0.99              # factor de descuento (ajustar según el entorno)

# Opciones de entrenamiento
batch_size = 64
num_epochs = 100  # número de épocas de entrenamiento (pasadas completas por el dataset)
target_update_interval = 10  # cada cuántas épocas sincronizar la red target con la principal

# == Entrenamiento FQE ==
for epoch in range(num_epochs):
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
        q_values = q_net(batch_states)                        # shape: (batch, num_actions)
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
                ns_np = ns.numpy()
                a_prime = policy_action(ns_np)       # acción según la política PPO
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

    # (Opcional) Mostrar la pérdida media de la época para monitorear convergencia
    if (epoch + 1) % 10 == 0 or epoch == 0:
        avg_loss = np.mean(batch_losses)
        print(f"Época {epoch+1}/{num_epochs} - Pérdida promedio: {avg_loss:.4f}")

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
    a = policy_action(s.numpy())  # acción (entera) de la política PPO
    # Calculamos Q(s, a) con nuestra red entrenada
    q_value = q_net(s.unsqueeze(0))  # hacemos forward con tamaño de batch 1
    q_value_sa = q_value[0, a].item()
    values.append(q_value_sa)

if values:
    estimated_value = np.mean(values)
    print(f"\nValor esperado estimado de la política PPO (retorno promedio inicial): {estimated_value:.3f}")
else:
    print("No se encontraron estados iniciales para evaluar la política.")
```

En este código, hemos implementado paso a paso el entrenamiento FQE y la evaluación de la política:

* **Preparación de datos:** convertimos el DataFrame de episodios en tensores PyTorch (`states`, `actions`, etc.). Suponemos que cada estado `state` es una representación numérica (por ejemplo, vector de observaciones). El espacio de acciones es discreto, identificado por enteros.
* **Modelo FQE:** definimos `QNetwork`, una red neuronal simple con dos capas ocultas (activación ReLU), que toma un estado como entrada y produce un vector de $Q$-valores para cada acción posible. Esta red estima $Q^\pi(s,a)$. Usamos dos instancias: `q_net` (la que entrenamos) y `target_net` (congelada, para proporcionar objetivos estables). Inicialmente, `target_net` copia los pesos de `q_net`.
* **Bucle de entrenamiento:** iteramos por `num_epochs`. En cada época barajamos los datos y los procesamos en mini-lotes (`batch_size`). Para cada batch, calculamos primero la predicción $Q(s,a)$ de la red actual. Luego calculamos el **target** para cada transición según la actualización de Bellman: si `done` es verdadero (1), usamos simplemente la recompensa (`target = r`); si no, usamos `target = r + \gamma \, Q_{\text{target}}(s', \pi(s'))`. Aquí es donde **integra la política PPO**: llamamos a `policy_action(s')` para obtener la acción $a'$ que la política PPO tomaría en el siguiente estado, y usamos $Q\_{\text{target}}(s', a')$ como estimación de $V^\pi(s')$. Esta implementación corresponde al término $\mathbb{E}\_{a' \sim \pi(\cdot|s')}Q(s',a')$ aproximado con una muestra (o decisión) de la política. Calculamos la pérdida MSE entre $Q(s,a)$ predicho por la red y el target calculado, y hacemos retropropagación para ajustar los pesos. Periódicamente sincronizamos `target_net` con `q_net` (cada 10 épocas en el ejemplo) para asegurarnos de que la estimación del objetivo se actualiza gradualmente (esto estabiliza el entrenamiento, de forma similar a **DQN**).
* **Evaluación de la política:** tras el entrenamiento, usamos la red `q_net` entrenada para estimar el valor de la política PPO. Identificamos los estados iniciales de cada episodio en el conjunto de datos (asumiendo que `df` está ordenado por episodio y marca el final con `done=True`). Para cada estado inicial, consultamos la política PPO (`policy_action`) para obtener la acción inicial $a\_0$, luego evaluamos $\hat{Q}^\pi(s\_0, a\_0)$ con la red. Finalmente, promediamos estos valores para obtener una estimación del **retorno inicial esperado** de la política PPO (esto aproxima $\eta^\pi$, asumiendo que la distribución de estados iniciales en los datos corresponde a $\mathbb{G}$). Si se conoce un único estado inicial determinístico, también podríamos simplemente calcular $Q(s\_{\text{init}}, \pi(s\_{\text{init}}))$ como estimación del valor.

**Notas finales:** El código presentado está en español y bien comentado para clarificar cada paso. En un caso real, se podría mejorar la integración con la política PPO para obtener directamente la distribución de acciones $\pi(a|s)$ y calcular expectativas exactas. Sin embargo, el esquema mostrado ilustra claramente cómo **FQE combina datos offline y la política objetivo PPO** para evaluar el valor esperado de dicha política de forma directa, cumpliendo con los requisitos del método directo de evaluación de políticas.

**Referencias:** Direct Method y FQE en OPE; Definición de $Q^\pi$ y valor de política; Ecuación de Bellman para política $\pi$; Algoritmo iterativo de FQE; Integración de la política en la actualización FQE.
