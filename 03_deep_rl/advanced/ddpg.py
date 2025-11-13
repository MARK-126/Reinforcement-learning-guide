"""
Deep Deterministic Policy Gradient (DDPG)
==========================================

Implementación de DDPG, un algoritmo actor-critic para control continuo.
DDPG extiende DQN a espacios de acción continuos usando una política determinista.

DDPG combina:
1. Actor determinista: μ_θ(s) que mapea estados a acciones
2. Critic Q-learning: Q_φ(s,a) que evalúa pares estado-acción
3. Experience replay: buffer para romper correlación temporal
4. Target networks: redes lentas para estabilidad
5. Exploration noise: Ornstein-Uhlenbeck o Gaussian

Actor update:
∇_θ J = E[∇_a Q(s,a)|_{a=μ(s)} ∇_θ μ_θ(s)]

Critic update (Bellman):
L = E[(Q(s,a) - (r + γQ'(s',μ'(s'))))²]

donde Q' y μ' son las target networks.

Características:
- Off-policy: aprende de buffer replay
- Deterministic: política determinista + ruido para exploración
- Continuous: diseñado para acciones continuas
- Sample efficient: reutiliza experiencias pasadas

Papers:
- "Continuous Control with Deep RL" (Lillicrap et al., 2016)
- "Deterministic Policy Gradient" (Silver et al., 2014)

Autor: MARK-126
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from collections import deque, namedtuple
import random
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt


# Transición para replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck Process para exploración en espacios continuos.

    Genera ruido correlacionado temporalmente, útil para exploración
    en problemas con inercia física (ej: robótica).

    dX_t = θ(μ - X_t)dt + σdW_t

    Parámetros:
    -----------
    size : int
        Dimensión del ruido (action_dim)
    mu : float
        Media de largo plazo
    theta : float
        Velocidad de reversión a la media
    sigma : float
        Volatilidad del proceso
    dt : float
        Paso de tiempo
    """

    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15,
                 sigma: float = 0.2, dt: float = 1e-2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        """Reinicia el estado del proceso."""
        self.state = np.ones(self.size) * self.mu

    def sample(self) -> np.ndarray:
        """
        Genera una muestra de ruido.

        Returns:
            noise: Vector de ruido [action_dim]
        """
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """
    Experience Replay Buffer.

    Almacena transiciones (s, a, r, s', done) y permite muestreo aleatorio.
    Rompe la correlación temporal de las experiencias.

    Parámetros:
    -----------
    capacity : int
        Capacidad máxima del buffer
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool):
        """Añade una transición al buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """Muestrea un batch aleatorio de transiciones."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class Actor(nn.Module):
    """
    Red Actor: Política determinista μ_θ(s).

    Mapea estados a acciones de manera determinista.
    Salida se pasa por tanh para limitar a [-1, 1].

    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    hidden_dims : List[int]
        Dimensiones de capas ocultas
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [400, 300]):
        super(Actor, self).__init__()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, action_dim)

        # Inicialización de pesos (como en paper)
        self.action_head.weight.data.uniform_(-3e-3, 3e-3)
        self.action_head.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: estado -> acción.

        Args:
            state: Estados [batch_size, state_dim]

        Returns:
            action: Acciones [batch_size, action_dim] en [-1, 1]
        """
        x = self.network(state)
        action = torch.tanh(self.action_head(x))
        return action


class Critic(nn.Module):
    """
    Red Critic: Q-function Q_φ(s, a).

    Estima el valor de un par estado-acción.
    La acción se concatena con el estado después de la primera capa.

    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    hidden_dims : List[int]
        Dimensiones de capas ocultas
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [400, 300]):
        super(Critic, self).__init__()

        # Primera capa procesa solo el estado
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])

        # Capas subsecuentes procesan estado + acción
        layers = []
        prev_dim = hidden_dims[0] + action_dim
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.q_head = nn.Linear(prev_dim, 1)

        # Inicialización de pesos
        self.q_head.weight.data.uniform_(-3e-3, 3e-3)
        self.q_head.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (estado, acción) -> Q-value.

        Args:
            state: Estados [batch_size, state_dim]
            action: Acciones [batch_size, action_dim]

        Returns:
            q_value: Q-values [batch_size]
        """
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = self.network(x)
        q_value = self.q_head(x).squeeze(-1)
        return q_value


class DDPGAgent:
    """
    Agente Deep Deterministic Policy Gradient (DDPG).

    DDPG es un algoritmo actor-critic off-policy para control continuo.
    Usa política determinista + ruido para exploración.

    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    actor_lr : float
        Tasa de aprendizaje del actor
    critic_lr : float
        Tasa de aprendizaje del critic
    gamma : float
        Factor de descuento
    tau : float
        Factor para soft update de target networks
    buffer_size : int
        Tamaño del replay buffer
    batch_size : int
        Tamaño del mini-batch
    noise_type : str
        Tipo de ruido: 'ou' (Ornstein-Uhlenbeck) o 'gaussian'
    noise_std : float
        Desviación estándar del ruido
    hidden_dims : List[int]
        Dimensiones de capas ocultas
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.001,
        buffer_size: int = 100000,
        batch_size: int = 64,
        noise_type: str = 'ou',
        noise_std: float = 0.2,
        hidden_dims: List[int] = [400, 300]
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Actor networks
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic networks
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Exploration noise
        self.noise_type = noise_type
        if noise_type == 'ou':
            self.noise = OrnsteinUhlenbeckNoise(action_dim, sigma=noise_std)
        else:  # gaussian
            self.noise_std = noise_std

        # Historial
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'q_values': []
        }

    def get_action(self, state: np.ndarray, noise_scale: float = 1.0,
                   training: bool = True) -> np.ndarray:
        """
        Selecciona acción usando política determinista + ruido.

        Args:
            state: Estado actual
            noise_scale: Escala del ruido (0 = sin ruido, 1 = ruido completo)
            training: Si False, sin ruido

        Returns:
            action: Acción seleccionada [action_dim]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()

        # Añadir ruido para exploración
        if training and noise_scale > 0:
            if self.noise_type == 'ou':
                noise = self.noise.sample()
            else:  # gaussian
                noise = np.random.randn(self.action_dim) * self.noise_std

            action = action + noise_scale * noise
            action = np.clip(action, -1, 1)

        return action

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Almacena transición en replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Realiza un paso de entrenamiento DDPG.

        1. Muestrea mini-batch del replay buffer
        2. Actualiza Critic minimizando TD error
        3. Actualiza Actor maximizando Q(s, μ(s))
        4. Soft update de target networks

        Returns:
            metrics: Diccionario con métricas (None si buffer insuficiente)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Muestrear batch
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convertir a tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.FloatTensor(np.array(batch.action)).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # ========== Actualizar Critic ==========
        with torch.no_grad():
            # Target: y = r + γQ'(s', μ'(s'))
            next_actions = self.actor_target(next_state_batch)
            target_q = self.critic_target(next_state_batch, next_actions)
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q

        # Q actual
        current_q = self.critic(state_batch, action_batch)

        # Critic loss: MSE
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ========== Actualizar Actor ==========
        # Actor loss: -E[Q(s, μ(s))]
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ========== Soft update de target networks ==========
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_q': current_q.mean().item()
        }

    def soft_update(self, source: nn.Module, target: nn.Module):
        """
        Soft update de target network.

        θ_target = τ*θ_source + (1-τ)*θ_target

        Args:
            source: Red fuente
            target: Red target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def train(
        self,
        env: gym.Env,
        n_episodes: int = 500,
        max_steps: int = 1000,
        warmup_steps: int = 1000,
        noise_decay: float = 0.9999,
        min_noise: float = 0.1,
        print_every: int = 10,
        save_every: int = 100,
        save_path: str = "ddpg_model.pth"
    ) -> Dict[str, List]:
        """
        Entrena el agente DDPG.

        Args:
            env: Entorno gymnasium
            n_episodes: Número de episodios
            max_steps: Pasos máximos por episodio
            warmup_steps: Steps de exploración aleatoria inicial
            noise_decay: Factor de decaimiento del ruido
            min_noise: Ruido mínimo
            print_every: Frecuencia de logging
            save_every: Frecuencia de guardado
            save_path: Ruta para guardar modelo

        Returns:
            history: Historial de entrenamiento
        """
        print(f"Entrenando DDPG...")
        print(f"Episodios: {n_episodes}, Warmup: {warmup_steps}")
        print(f"Noise: {self.noise_type}, Gamma: {self.gamma}, Tau: {self.tau}\n")

        noise_scale = 1.0
        total_steps = 0

        for episode in range(n_episodes):
            state, _ = env.reset()
            if self.noise_type == 'ou':
                self.noise.reset()

            episode_reward = 0
            episode_metrics = []

            for step in range(max_steps):
                # Warmup: exploración aleatoria
                if total_steps < warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = self.get_action(state, noise_scale, training=True)

                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Almacenar transición
                self.store_transition(state, action, reward, next_state, done)

                # Entrenar (después de warmup)
                if total_steps >= warmup_steps:
                    metrics = self.train_step()
                    if metrics is not None:
                        episode_metrics.append(metrics)

                episode_reward += reward
                total_steps += 1
                state = next_state

                if done:
                    break

            # Decay noise
            if total_steps >= warmup_steps:
                noise_scale = max(min_noise, noise_scale * noise_decay)

            # Registrar episodio
            self.history['episode_rewards'].append(episode_reward)
            self.history['episode_lengths'].append(step + 1)

            if episode_metrics:
                avg_metrics = {
                    key: np.mean([m[key] for m in episode_metrics])
                    for key in episode_metrics[0].keys()
                }
                self.history['actor_losses'].append(avg_metrics['actor_loss'])
                self.history['critic_losses'].append(avg_metrics['critic_loss'])
                self.history['q_values'].append(avg_metrics['mean_q'])

            # Logging
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.history['episode_rewards'][-100:])
                avg_length = np.mean(self.history['episode_lengths'][-100:])
                avg_q = np.mean(self.history['q_values'][-100:]) if self.history['q_values'] else 0

                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg(100): {avg_reward:.2f} | "
                      f"Length: {step + 1} | "
                      f"Noise: {noise_scale:.3f} | "
                      f"Q: {avg_q:.2f}")

            # Guardar modelo
            if save_every > 0 and (episode + 1) % save_every == 0:
                self.save(save_path)

        print("\nEntrenamiento completado!")
        return self.history

    def save(self, path: str):
        """Guarda el modelo."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"Modelo guardado en {path}")

    def load(self, path: str):
        """Carga el modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.history = checkpoint['history']
        print(f"Modelo cargado desde {path}")


def evaluate_agent(agent: DDPGAgent, env: gym.Env,
                   n_episodes: int = 10, render: bool = False) -> Tuple[float, float]:
    """
    Evalúa el agente entrenado.

    Args:
        agent: Agente DDPG
        env: Entorno
        n_episodes: Número de episodios de evaluación
        render: Si True, renderiza

    Returns:
        mean_reward: Recompensa promedio
        std_reward: Desviación estándar
    """
    rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, noise_scale=0.0, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if render:
                env.render()

        rewards.append(episode_reward)

    return np.mean(rewards), np.std(rewards)


def plot_training_results(history: Dict[str, List], save_path: str = 'ddpg_training.png'):
    """
    Visualiza resultados del entrenamiento.

    Args:
        history: Historial de entrenamiento
        save_path: Ruta donde guardar la figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Recompensas
    ax = axes[0, 0]
    rewards = history['episode_rewards']
    ax.plot(rewards, alpha=0.3, label='Episode Reward')
    if len(rewards) >= 10:
        window = min(100, len(rewards) // 10)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), moving_avg,
               label=f'MA({window})', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Losses
    ax = axes[0, 1]
    if history['actor_losses']:
        ax.plot(history['actor_losses'], alpha=0.6, label='Actor Loss')
        ax.plot(history['critic_losses'], alpha=0.6, label='Critic Loss')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Actor & Critic Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-values
    ax = axes[1, 0]
    if history['q_values']:
        q_vals = history['q_values']
        ax.plot(q_vals, alpha=0.6, label='Mean Q-value')
        if len(q_vals) >= 10:
            window = min(100, len(q_vals) // 10)
            moving_avg = np.convolve(q_vals, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(q_vals)), moving_avg,
                   label=f'MA({window})', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q-value')
    ax.set_title('Mean Q-values')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode Lengths
    ax = axes[1, 1]
    if history['episode_lengths']:
        lengths = history['episode_lengths']
        ax.plot(lengths, alpha=0.3, label='Episode Length')
        if len(lengths) >= 10:
            window = min(100, len(lengths) // 10)
            moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(lengths)), moving_avg,
                   label=f'MA({window})', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length')
    ax.set_title('Episode Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráficos guardados en '{save_path}'")


# ==================== MAIN ====================

def main():
    """Ejemplos de uso: Entrenar DDPG en ambientes continuos"""

    print("=" * 70)
    print("DEEP DETERMINISTIC POLICY GRADIENT (DDPG)")
    print("=" * 70)

    # ========== Ejemplo 1: Pendulum ==========
    print("\n" + "=" * 70)
    print("Ejemplo 1: Pendulum-v1")
    print("=" * 70)

    env_pendulum = gym.make('Pendulum-v1')
    state_dim = env_pendulum.observation_space.shape[0]
    action_dim = env_pendulum.action_space.shape[0]

    print(f"\nEntorno: Pendulum-v1")
    print(f"Estado: {state_dim} dimensiones")
    print(f"Acciones: {action_dim} dimensiones (continuas)\n")

    agent_pendulum = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.001,
        buffer_size=100000,
        batch_size=64,
        noise_type='ou',
        noise_std=0.2,
        hidden_dims=[400, 300]
    )

    history_pendulum = agent_pendulum.train(
        env=env_pendulum,
        n_episodes=200,
        max_steps=200,
        warmup_steps=1000,
        noise_decay=0.9995,
        min_noise=0.1,
        print_every=10,
        save_every=50,
        save_path='ddpg_pendulum.pth'
    )

    plot_training_results(history_pendulum, 'ddpg_pendulum_training.png')

    print("\nEvaluando política aprendida...")
    mean_reward, std_reward = evaluate_agent(agent_pendulum, env_pendulum, n_episodes=50)
    print(f"Recompensa promedio: {mean_reward:.2f} ± {std_reward:.2f}")

    env_pendulum.close()

    # ========== Ejemplo 2: MountainCarContinuous ==========
    print("\n" + "=" * 70)
    print("Ejemplo 2: MountainCarContinuous-v0")
    print("=" * 70)

    env_mountain = gym.make('MountainCarContinuous-v0')
    state_dim_m = env_mountain.observation_space.shape[0]
    action_dim_m = env_mountain.action_space.shape[0]

    print(f"\nEntorno: MountainCarContinuous-v0")
    print(f"Estado: {state_dim_m} dimensiones")
    print(f"Acciones: {action_dim_m} dimensiones (continuas)\n")

    agent_mountain = DDPGAgent(
        state_dim=state_dim_m,
        action_dim=action_dim_m,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.001,
        buffer_size=50000,
        batch_size=64,
        noise_type='gaussian',
        noise_std=0.3,
        hidden_dims=[400, 300]
    )

    history_mountain = agent_mountain.train(
        env=env_mountain,
        n_episodes=500,
        max_steps=999,
        warmup_steps=2000,
        noise_decay=0.999,
        min_noise=0.05,
        print_every=20,
        save_every=100,
        save_path='ddpg_mountain.pth'
    )

    plot_training_results(history_mountain, 'ddpg_mountain_training.png')

    print("\nEvaluando política aprendida...")
    mean_reward_m, std_reward_m = evaluate_agent(agent_mountain, env_mountain, n_episodes=30)
    print(f"Recompensa promedio: {mean_reward_m:.2f} ± {std_reward_m:.2f}")

    env_mountain.close()

    print("\n" + "=" * 70)
    print("CARACTERÍSTICAS DE DDPG")
    print("=" * 70)
    print("1. Continuous control - diseñado para acciones continuas")
    print("2. Deterministic policy - política determinista + ruido")
    print("3. Off-policy - sample efficient con replay buffer")
    print("4. Actor-Critic - combina policy gradient con Q-learning")
    print("5. Target networks - estabiliza entrenamiento")
    print("\nLimitaciones:")
    print("- Sensible a hiperparámetros")
    print("- Puede ser inestable en algunos ambientes")
    print("- Superado por TD3 y SAC en muchos casos")
    print("=" * 70)


if __name__ == "__main__":
    main()
