"""
Dueling Deep Q-Network (Dueling DQN) Implementation

Dueling DQN separa la estimación de Q-values en dos streams:
- Value stream V(s): Valor de estar en el estado s
- Advantage stream A(s,a): Ventaja de tomar la acción a en el estado s

Combinación: Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))

Esta arquitectura permite aprender qué estados son valiosos sin necesidad
de aprender el efecto de cada acción para cada estado. Especialmente útil
cuando muchas acciones no afectan el estado de manera relevante.

Paper: "Dueling Network Architectures for Deep Reinforcement Learning"
       (Wang et al., 2016)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import os


# Transición para replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    Experience Replay Buffer

    Almacena transiciones (s, a, r, s', done) y permite muestreo aleatorio.
    Esto rompe la correlación temporal de las experiencias.

    Args:
        capacity: Capacidad máxima del buffer
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Añade una transición al buffer"""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """Muestrea un batch aleatorio de transiciones"""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DuelingDQN(nn.Module):
    """
    Red Neuronal Dueling DQN

    Arquitectura con dos streams separados:
    1. Feature extraction: Capas compartidas que extraen features del estado
    2. Value stream: Estima V(s)
    3. Advantage stream: Estima A(s,a) para cada acción
    4. Aggregation: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

    Args:
        state_dim: Dimensión del espacio de estados
        action_dim: Número de acciones discretas
        hidden_dim: Dimensión de capas ocultas
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DuelingDQN, self).__init__()

        self.action_dim = action_dim

        # Feature extraction (capas compartidas)
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: estado -> Q-values

        Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))

        Args:
            x: Tensor de estados [batch_size, state_dim]

        Returns:
            Q-values [batch_size, action_dim]
        """
        # Features compartidas
        features = self.feature(x)

        # Value stream: V(s) - un solo valor por estado
        value = self.value_stream(features)

        # Advantage stream: A(s,a) - un valor por acción
        advantage = self.advantage_stream(features)

        # Aggregation: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # Restamos la media para garantizar identificabilidad única
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def get_value_and_advantage(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna V(s) y A(s,a) por separado (útil para análisis)

        Args:
            x: Tensor de estados

        Returns:
            value: V(s)
            advantage: A(s,a)
        """
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value, advantage


class DuelingDQNAgent:
    """
    Agente Dueling DQN

    Usa arquitectura Dueling para mejorar el aprendizaje de Q-values.
    Puede combinarse con Double DQN para mejores resultados.

    Args:
        state_dim: Dimensión del espacio de estados
        action_dim: Número de acciones discretas
        learning_rate: Tasa de aprendizaje (default: 1e-3)
        gamma: Factor de descuento (default: 0.99)
        epsilon_start: Epsilon inicial para exploración (default: 1.0)
        epsilon_end: Epsilon final (default: 0.01)
        epsilon_decay: Pasos para decaer epsilon (default: 1000)
        buffer_size: Tamaño del replay buffer (default: 10000)
        batch_size: Tamaño del mini-batch (default: 64)
        target_update: Frecuencia de actualización de target network (default: 10)
        tau: Factor para soft update (None = hard update) (default: None)
        hidden_dim: Dimensión de capas ocultas (default: 128)
        use_double_dqn: Si True, usa Double DQN update (default: True)
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 1000,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update: int = 10,
                 tau: Optional[float] = None,
                 hidden_dim: int = 128,
                 use_double_dqn: bool = True):

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.tau = tau
        self.use_double_dqn = use_double_dqn

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Dueling Q-Network (online) y Target Network
        self.q_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Contadores
        self.steps = 0
        self.episodes = 0

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Selecciona acción usando ε-greedy

        Args:
            state: Estado actual (numpy array)
            training: Si False, usa política greedy (ε=0)

        Returns:
            Acción seleccionada (int)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def update_epsilon(self) -> None:
        """Actualiza epsilon con decaimiento exponencial"""
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-self.steps / self.epsilon_decay)

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Almacena transición en replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Realiza un paso de entrenamiento

        Puede usar DQN estándar o Double DQN según configuración.

        Dueling DQN + Double DQN:
        1. Muestrea mini-batch
        2. Calcula Q-values actuales con arquitectura Dueling
        3. Si Double DQN: selecciona con online, evalúa con target
        4. Si DQN estándar: max sobre target network
        5. Actualiza red

        Returns:
            Loss del paso de entrenamiento (None si buffer insuficiente)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Muestrear batch
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convertir a tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # Q-values actuales: Q(s, a) usando arquitectura Dueling
        current_q_values = self.q_network(state_batch).gather(1, action_batch)

        # Calcular targets
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: seleccionar con online, evaluar con target
                next_actions = self.q_network(next_state_batch).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_state_batch).gather(1, next_actions).squeeze()
            else:
                # DQN estándar: max sobre target network
                next_q_values = self.target_network(next_state_batch).max(1)[0]

            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Loss: MSE entre Q actual y target
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self) -> None:
        """
        Actualiza target network

        - Hard update: Copia completa de pesos
        - Soft update: θ_target = τ*θ_online + (1-τ)*θ_target
        """
        if self.tau is None:
            # Hard update
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # Soft update
            for target_param, online_param in zip(self.target_network.parameters(),
                                                   self.q_network.parameters()):
                target_param.data.copy_(
                    self.tau * online_param.data + (1.0 - self.tau) * target_param.data
                )

    def analyze_value_advantage(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analiza V(s) y A(s,a) para un estado dado

        Útil para entender qué está aprendiendo la red.

        Args:
            state: Estado a analizar

        Returns:
            Dictionary con 'value', 'advantage', 'q_values'
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value, advantage = self.q_network.get_value_and_advantage(state_tensor)
            q_values = self.q_network(state_tensor)

            return {
                'value': value.cpu().numpy()[0, 0],
                'advantage': advantage.cpu().numpy()[0],
                'q_values': q_values.cpu().numpy()[0]
            }

    def save(self, path: str) -> None:
        """
        Guarda el modelo y parámetros

        Args:
            path: Ruta del archivo para guardar
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'use_double_dqn': self.use_double_dqn
        }, path)
        print(f"Modelo guardado en {path}")

    def load(self, path: str) -> None:
        """
        Carga el modelo y parámetros

        Args:
            path: Ruta del archivo a cargar
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.use_double_dqn = checkpoint.get('use_double_dqn', True)
        print(f"Modelo cargado desde {path}")


def train_dueling_dqn(env: gym.Env,
                      agent: DuelingDQNAgent,
                      n_episodes: int = 500,
                      max_steps: int = 500,
                      save_every: int = 100,
                      save_path: str = "dueling_dqn_model.pth") -> Tuple[List[float], List[float]]:
    """
    Entrena agente Dueling DQN

    Args:
        env: Ambiente gymnasium
        agent: DuelingDQNAgent
        n_episodes: Número de episodios
        max_steps: Máximo de steps por episodio
        save_every: Frecuencia de guardado del modelo (episodios)
        save_path: Ruta para guardar el modelo

    Returns:
        rewards_history: Lista de recompensas por episodio
        losses_history: Lista de losses promedio por episodio
    """
    rewards_history = []
    losses_history = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_losses = []

        for step in range(max_steps):
            # Seleccionar acción
            action = agent.get_action(state, training=True)

            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Almacenar transición
            agent.store_transition(state, action, reward, next_state, done)

            # Entrenar
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            # Actualizar epsilon y steps
            agent.steps += 1
            agent.update_epsilon()

            # Soft update cada step (si tau no es None)
            if agent.tau is not None:
                agent.update_target_network()

            episode_reward += reward
            state = next_state

            if done:
                break

        # Hard update periódicamente (si tau es None)
        if agent.tau is None and episode % agent.target_update == 0:
            agent.update_target_network()

        agent.episodes += 1
        rewards_history.append(episode_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses_history.append(avg_loss)

        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}/{n_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Loss: {avg_loss:.4f}")

        # Guardar modelo periódicamente
        if save_every > 0 and (episode + 1) % save_every == 0:
            agent.save(save_path)

    return rewards_history, losses_history


def evaluate_agent(env: gym.Env,
                   agent: DuelingDQNAgent,
                   n_episodes: int = 10,
                   render: bool = False,
                   analyze: bool = False) -> float:
    """
    Evalúa el agente sin exploración

    Args:
        env: Ambiente gymnasium
        agent: DuelingDQNAgent
        n_episodes: Número de episodios de evaluación
        render: Si True, renderiza el ambiente
        analyze: Si True, muestra análisis de V(s) y A(s,a)

    Returns:
        Recompensa promedio
    """
    eval_rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            if render:
                env.render()

            # Analizar primer estado del primer episodio
            if analyze and ep == 0 and step == 0:
                analysis = agent.analyze_value_advantage(state)
                print("\nAnálisis del estado inicial:")
                print(f"V(s) = {analysis['value']:.3f}")
                print(f"A(s,a) = {analysis['advantage']}")
                print(f"Q(s,a) = {analysis['q_values']}")
                print(f"Mejor acción: {np.argmax(analysis['q_values'])}\n")

            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state
            step += 1

        eval_rewards.append(episode_reward)

    return np.mean(eval_rewards)


def plot_training(rewards: List[float],
                 losses: List[float],
                 save_path: str = "dueling_dqn_training.png") -> None:
    """
    Visualiza resultados del entrenamiento

    Args:
        rewards: Lista de recompensas por episodio
        losses: Lista de losses por episodio
        save_path: Ruta para guardar la figura
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Recompensas
    ax1.plot(rewards, alpha=0.6, label='Reward')
    window = 10
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg,
                label=f'MA({window})', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Dueling DQN Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Losses
    ax2.plot(losses, alpha=0.6, label='Loss')
    if len(losses) >= window:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(losses)), moving_avg,
                label=f'MA({window})', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Dueling DQN Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráficos guardados en '{save_path}'")


def visualize_value_advantage(agent: DuelingDQNAgent,
                              env: gym.Env,
                              n_samples: int = 100,
                              save_path: str = "value_advantage_analysis.png") -> None:
    """
    Visualiza la distribución de V(s) y A(s,a) durante la evaluación

    Args:
        agent: DuelingDQNAgent entrenado
        env: Ambiente gymnasium
        n_samples: Número de estados a muestrear
        save_path: Ruta para guardar la figura
    """
    values = []
    advantages_list = []

    # Recolectar muestras
    for _ in range(n_samples // 10):  # Varios episodios
        state, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 10:  # Hasta 10 pasos por episodio
            analysis = agent.analyze_value_advantage(state)
            values.append(analysis['value'])
            advantages_list.append(analysis['advantage'])

            action = agent.get_action(state, training=False)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

    values = np.array(values)
    advantages = np.array(advantages_list)

    # Visualizar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Distribución de valores
    ax1.hist(values, bins=30, alpha=0.7, edgecolor='black')
    ax1.axvline(values.mean(), color='red', linestyle='--',
                label=f'Mean: {values.mean():.2f}')
    ax1.set_xlabel('V(s)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Value Stream Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Ventajas por acción
    for action in range(advantages.shape[1]):
        ax2.hist(advantages[:, action], bins=20, alpha=0.5,
                label=f'Action {action}')
    ax2.set_xlabel('A(s,a)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Advantage Stream Distribution per Action')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Análisis V/A guardado en '{save_path}'")


def main():
    """
    Ejemplo de uso: Entrenar Dueling DQN en LunarLander

    Dueling DQN mejora el aprendizaje al separar valor de estado y ventajas de acciones.
    Se combina con Double DQN para mejores resultados.
    """
    # Seleccionar ambiente
    # env_name = 'CartPole-v1'  # Más simple
    env_name = 'LunarLander-v2'  # Más complejo (recomendado para Dueling)

    print(f"=== Entrenando Dueling DQN en {env_name} ===\n")

    # Crear ambiente
    env = gym.make(env_name)

    # Obtener dimensiones
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Estado: {state_dim} dimensiones")
    print(f"Acciones: {action_dim}")
    print()

    # Configuración según ambiente
    if env_name == 'CartPole-v1':
        config = {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 500,
            'buffer_size': 10000,
            'batch_size': 64,
            'target_update': 10,
            'tau': None,  # Hard update
            'hidden_dim': 128,
            'use_double_dqn': True,
            'n_episodes': 300,
            'max_steps': 500
        }
    else:  # LunarLander-v2
        config = {
            'learning_rate': 5e-4,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 1000,
            'buffer_size': 50000,
            'batch_size': 128,
            'target_update': 5,
            'tau': 0.005,  # Soft update
            'hidden_dim': 256,
            'use_double_dqn': True,  # Combinar con Double DQN
            'n_episodes': 600,
            'max_steps': 1000
        }

    # Crear agente
    agent = DuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **{k: v for k, v in config.items() if k not in ['n_episodes', 'max_steps']}
    )

    print("Arquitectura Dueling:")
    print(agent.q_network)
    print()

    # Entrenar
    update_type = 'Double DQN' if config['use_double_dqn'] else 'Standard DQN'
    update_freq = f"Soft update (tau={config['tau']})" if config['tau'] else 'Hard update'
    print(f"Entrenando Dueling DQN + {update_type} durante {config['n_episodes']} episodios...")
    print(f"Update strategy: {update_freq}")
    print()

    rewards, losses = train_dueling_dqn(
        env, agent,
        n_episodes=config['n_episodes'],
        max_steps=config['max_steps'],
        save_every=100,
        save_path=f"dueling_dqn_{env_name.lower()}.pth"
    )

    # Visualizar entrenamiento
    plot_training(rewards, losses, save_path=f"dueling_dqn_{env_name.lower()}_training.png")

    # Evaluación final
    print("\n" + "="*50)
    print("Evaluación final (sin exploración):")
    avg_reward = evaluate_agent(env, agent, n_episodes=20, analyze=True)
    print(f"Recompensa promedio: {avg_reward:.2f}")

    print(f"\nReward promedio últimos 10 episodios entrenamiento: {np.mean(rewards[-10:]):.2f}")
    print(f"Reward promedio últimos 50 episodios entrenamiento: {np.mean(rewards[-50:]):.2f}")

    # Análisis de value y advantage streams
    print("\nGenerando análisis de Value/Advantage streams...")
    visualize_value_advantage(
        agent, env, n_samples=100,
        save_path=f"dueling_dqn_{env_name.lower()}_va_analysis.png"
    )

    # Guardar modelo final
    agent.save(f"dueling_dqn_{env_name.lower()}_final.pth")

    env.close()

    print("\n" + "="*50)
    print("Ventajas de Dueling DQN:")
    print("1. Separa valor de estado de ventajas de acciones")
    print("2. Aprende más eficientemente qué estados son valiosos")
    print("3. Mejor generalización en espacios de acción grandes")
    print("4. Útil cuando muchas acciones tienen efecto similar")
    print("5. Se combina perfectamente con Double DQN")
    print("\nCuándo usar Dueling DQN:")
    print("- Ambientes donde el valor del estado es importante")
    print("- Muchas acciones disponibles pero pocas son relevantes")
    print("- Cuando necesitas mejor generalización")


if __name__ == "__main__":
    main()
