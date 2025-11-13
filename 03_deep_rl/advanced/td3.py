"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
======================================================

Implementación de TD3, una mejora significativa sobre DDPG que resuelve
problemas de sobreestimación y inestabilidad.

TD3 introduce tres innovaciones clave sobre DDPG:

1. **Twin Q-networks (Clipped Double Q-learning)**:
   - Usa dos critics Q₁ y Q₂
   - Target value = r + γ min(Q₁', Q₂')(s', a')
   - Reduce sobreestimación de Q-values

2. **Delayed Policy Updates**:
   - Actualiza actor cada d steps
   - Actualiza critics más frecuentemente
   - Reduce varianza del gradiente de política

3. **Target Policy Smoothing**:
   - Añade ruido a acciones target
   - a' = clip(μ'(s') + ε, a_low, a_high)
   - Suaviza superficie de Q, más robusto

Objetivo del Actor:
∇_θ J = E[∇_a Q₁(s,a)|_{a=μ(s)} ∇_θ μ_θ(s)]

Objetivo de los Critics:
L_i = E[(Q_i(s,a) - y)²]
donde y = r + γ min_{i=1,2} Q_i'(s', μ'(s') + ε)

Papers:
- "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018)
- "Continuous Control with Deep RL" (Lillicrap et al., 2016)

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


class ReplayBuffer:
    """
    Experience Replay Buffer.

    Almacena transiciones (s, a, r, s', done) y permite muestreo aleatorio.

    Parámetros:
    -----------
    capacity : int
        Capacidad máxima del buffer
    """

    def __init__(self, capacity: int = 1000000):
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

    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    max_action : float
        Valor máximo de la acción
    hidden_dims : List[int]
        Dimensiones de capas ocultas
    """

    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 hidden_dims: List[int] = [256, 256]):
        super(Actor, self).__init__()

        self.max_action = max_action

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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: estado -> acción.

        Args:
            state: Estados [batch_size, state_dim]

        Returns:
            action: Acciones [batch_size, action_dim]
        """
        x = self.network(state)
        action = self.max_action * torch.tanh(self.action_head(x))
        return action


class Critic(nn.Module):
    """
    Red Twin Critic: Dos Q-functions Q₁_φ(s,a) y Q₂_φ(s,a).

    TD3 usa dos critics para reducir sobreestimación.

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
                 hidden_dims: List[int] = [256, 256]):
        super(Critic, self).__init__()

        # Q1
        layers1 = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers1.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers1.append(nn.Linear(prev_dim, 1))
        self.q1 = nn.Sequential(*layers1)

        # Q2
        layers2 = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers2.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers2.append(nn.Linear(prev_dim, 1))
        self.q2 = nn.Sequential(*layers2)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: (estado, acción) -> (Q1, Q2).

        Args:
            state: Estados [batch_size, state_dim]
            action: Acciones [batch_size, action_dim]

        Returns:
            q1: Q1-values [batch_size]
            q2: Q2-values [batch_size]
        """
        sa = torch.cat([state, action], dim=1)
        q1 = self.q1(sa).squeeze(-1)
        q2 = self.q2(sa).squeeze(-1)
        return q1, q2

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Retorna solo Q1 (usado para actualizar actor)."""
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa).squeeze(-1)


class TD3Agent:
    """
    Agente Twin Delayed Deep Deterministic Policy Gradient (TD3).

    TD3 mejora DDPG con:
    1. Twin critics para reducir sobreestimación
    2. Delayed policy updates para reducir varianza
    3. Target policy smoothing para robustez

    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    max_action : float
        Valor máximo de la acción
    actor_lr : float
        Tasa de aprendizaje del actor
    critic_lr : float
        Tasa de aprendizaje de los critics
    gamma : float
        Factor de descuento
    tau : float
        Factor para soft update de target networks
    policy_noise : float
        Ruido para target policy smoothing
    noise_clip : float
        Límite para el ruido de target policy
    policy_delay : int
        Frecuencia de actualización del actor (cada d steps)
    buffer_size : int
        Tamaño del replay buffer
    batch_size : int
        Tamaño del mini-batch
    exploration_noise : float
        Desviación estándar del ruido de exploración
    hidden_dims : List[int]
        Dimensiones de capas ocultas
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        exploration_noise: float = 0.1,
        hidden_dims: List[int] = [256, 256]
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dims).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin Critic networks
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Contador para delayed policy update
        self.total_it = 0

        # Historial
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'q1_values': [],
            'q2_values': []
        }

    def get_action(self, state: np.ndarray, noise_scale: float = 1.0,
                   training: bool = True) -> np.ndarray:
        """
        Selecciona acción usando política determinista + ruido gaussiano.

        Args:
            state: Estado actual
            noise_scale: Escala del ruido de exploración
            training: Si False, sin ruido

        Returns:
            action: Acción seleccionada [action_dim]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()

        # Añadir ruido gaussiano para exploración
        if training and noise_scale > 0:
            noise = np.random.randn(self.action_dim) * self.exploration_noise * noise_scale
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Almacena transición en replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Realiza un paso de entrenamiento TD3.

        Implementa las tres mejoras clave de TD3:
        1. Clipped Double Q-learning
        2. Delayed Policy Updates
        3. Target Policy Smoothing

        Returns:
            metrics: Diccionario con métricas (None si buffer insuficiente)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.total_it += 1

        # Muestrear batch
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convertir a tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.FloatTensor(np.array(batch.action)).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        with torch.no_grad():
            # ========== Target Policy Smoothing ==========
            # Añadir ruido a acción target (regularización)
            noise = (torch.randn_like(action_batch) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state_batch) + noise).clamp(
                -self.max_action, self.max_action
            )

            # ========== Clipped Double Q-learning ==========
            # Calcular Q-values target con ambos critics
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            # Tomar el mínimo para reducir sobreestimación
            target_q = torch.min(target_q1, target_q2)
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q

        # Q-values actuales
        current_q1, current_q2 = self.critic(state_batch, action_batch)

        # Critic loss: MSE para ambos critics
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimizar Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Métricas para retornar
        metrics = {
            'critic_loss': critic_loss.item(),
            'mean_q1': current_q1.mean().item(),
            'mean_q2': current_q2.mean().item()
        }

        # ========== Delayed Policy Updates ==========
        # Actualizar actor solo cada policy_delay steps
        if self.total_it % self.policy_delay == 0:
            # Actor loss: -E[Q1(s, μ(s))]
            actor_loss = -self.critic.q1_forward(state_batch, self.actor(state_batch)).mean()

            # Optimizar Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update de target networks
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)

            metrics['actor_loss'] = actor_loss.item()
        else:
            metrics['actor_loss'] = None

        return metrics

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
        n_episodes: int = 1000,
        max_steps: int = 1000,
        warmup_steps: int = 25000,
        noise_decay: float = 1.0,
        min_noise: float = 0.1,
        print_every: int = 10,
        save_every: int = 100,
        save_path: str = "td3_model.pth"
    ) -> Dict[str, List]:
        """
        Entrena el agente TD3.

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
        print(f"Entrenando TD3...")
        print(f"Episodios: {n_episodes}, Warmup: {warmup_steps}")
        print(f"Policy delay: {self.policy_delay}, Gamma: {self.gamma}, Tau: {self.tau}")
        print(f"Policy noise: {self.policy_noise}, Noise clip: {self.noise_clip}\n")

        noise_scale = 1.0
        total_steps = 0

        for episode in range(n_episodes):
            state, _ = env.reset()
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
                # Filtrar métricas de actor (pueden ser None en algunos steps)
                actor_losses = [m['actor_loss'] for m in episode_metrics if m['actor_loss'] is not None]
                critic_losses = [m['critic_loss'] for m in episode_metrics]
                q1_values = [m['mean_q1'] for m in episode_metrics]
                q2_values = [m['mean_q2'] for m in episode_metrics]

                if actor_losses:
                    self.history['actor_losses'].append(np.mean(actor_losses))
                if critic_losses:
                    self.history['critic_losses'].append(np.mean(critic_losses))
                if q1_values:
                    self.history['q1_values'].append(np.mean(q1_values))
                if q2_values:
                    self.history['q2_values'].append(np.mean(q2_values))

            # Logging
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.history['episode_rewards'][-100:])
                avg_length = np.mean(self.history['episode_lengths'][-100:])
                avg_q1 = np.mean(self.history['q1_values'][-100:]) if self.history['q1_values'] else 0
                avg_q2 = np.mean(self.history['q2_values'][-100:]) if self.history['q2_values'] else 0

                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg(100): {avg_reward:.2f} | "
                      f"Length: {step + 1} | "
                      f"Noise: {noise_scale:.3f} | "
                      f"Q1: {avg_q1:.2f} | Q2: {avg_q2:.2f}")

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
            'total_it': self.total_it,
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
        self.total_it = checkpoint['total_it']
        self.history = checkpoint['history']
        print(f"Modelo cargado desde {path}")


def evaluate_agent(agent: TD3Agent, env: gym.Env,
                   n_episodes: int = 10, render: bool = False) -> Tuple[float, float]:
    """
    Evalúa el agente entrenado.

    Args:
        agent: Agente TD3
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


def plot_training_results(history: Dict[str, List], save_path: str = 'td3_training.png'):
    """
    Visualiza resultados del entrenamiento.

    Args:
        history: Historial de entrenamiento
        save_path: Ruta donde guardar la figura
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

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
    if history['actor_losses'] and history['critic_losses']:
        ax.plot(history['actor_losses'], alpha=0.6, label='Actor Loss')
        ax.plot(history['critic_losses'], alpha=0.6, label='Critic Loss')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Actor & Critic Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-values (Twin Critics)
    ax = axes[0, 2]
    if history['q1_values'] and history['q2_values']:
        ax.plot(history['q1_values'], alpha=0.6, label='Q1')
        ax.plot(history['q2_values'], alpha=0.6, label='Q2')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q-value')
    ax.set_title('Twin Q-values')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-value difference (Q1 - Q2)
    ax = axes[1, 0]
    if history['q1_values'] and history['q2_values']:
        q_diff = np.array(history['q1_values']) - np.array(history['q2_values'])
        ax.plot(q_diff, alpha=0.6, label='Q1 - Q2')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q-value Difference')
    ax.set_title('Q1 - Q2 (should be close to 0)')
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

    # Reward distribution (últimos 100 episodios)
    ax = axes[1, 2]
    if len(rewards) >= 10:
        recent_rewards = rewards[-100:]
        ax.hist(recent_rewards, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(recent_rewards), color='r', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(recent_rewards):.2f}')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution (last 100 episodes)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráficos guardados en '{save_path}'")


# ==================== MAIN ====================

def main():
    """Ejemplos de uso: Entrenar TD3 en ambientes continuos"""

    print("=" * 70)
    print("TWIN DELAYED DEEP DETERMINISTIC POLICY GRADIENT (TD3)")
    print("=" * 70)

    # ========== Ejemplo 1: Pendulum ==========
    print("\n" + "=" * 70)
    print("Ejemplo 1: Pendulum-v1")
    print("=" * 70)

    env_pendulum = gym.make('Pendulum-v1')
    state_dim = env_pendulum.observation_space.shape[0]
    action_dim = env_pendulum.action_space.shape[0]
    max_action = float(env_pendulum.action_space.high[0])

    print(f"\nEntorno: Pendulum-v1")
    print(f"Estado: {state_dim} dimensiones")
    print(f"Acciones: {action_dim} dimensiones (continuas)")
    print(f"Max action: {max_action}\n")

    agent_pendulum = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        buffer_size=1000000,
        batch_size=256,
        exploration_noise=0.1,
        hidden_dims=[256, 256]
    )

    history_pendulum = agent_pendulum.train(
        env=env_pendulum,
        n_episodes=200,
        max_steps=200,
        warmup_steps=1000,
        noise_decay=0.999,
        min_noise=0.1,
        print_every=10,
        save_every=50,
        save_path='td3_pendulum.pth'
    )

    plot_training_results(history_pendulum, 'td3_pendulum_training.png')

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
    max_action_m = float(env_mountain.action_space.high[0])

    print(f"\nEntorno: MountainCarContinuous-v0")
    print(f"Estado: {state_dim_m} dimensiones")
    print(f"Acciones: {action_dim_m} dimensiones (continuas)")
    print(f"Max action: {max_action_m}\n")

    agent_mountain = TD3Agent(
        state_dim=state_dim_m,
        action_dim=action_dim_m,
        max_action=max_action_m,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        buffer_size=1000000,
        batch_size=256,
        exploration_noise=0.1,
        hidden_dims=[256, 256]
    )

    history_mountain = agent_mountain.train(
        env=env_mountain,
        n_episodes=500,
        max_steps=999,
        warmup_steps=5000,
        noise_decay=0.995,
        min_noise=0.05,
        print_every=20,
        save_every=100,
        save_path='td3_mountain.pth'
    )

    plot_training_results(history_mountain, 'td3_mountain_training.png')

    print("\nEvaluando política aprendida...")
    mean_reward_m, std_reward_m = evaluate_agent(agent_mountain, env_mountain, n_episodes=30)
    print(f"Recompensa promedio: {mean_reward_m:.2f} ± {std_reward_m:.2f}")

    env_mountain.close()

    print("\n" + "=" * 70)
    print("MEJORAS DE TD3 SOBRE DDPG")
    print("=" * 70)
    print("1. Twin Critics (Clipped Double Q-learning):")
    print("   - Reduce sobreestimación de Q-values")
    print("   - Usa min(Q1, Q2) para target values")
    print("\n2. Delayed Policy Updates:")
    print("   - Actor se actualiza menos frecuentemente que Critic")
    print("   - Reduce varianza del gradiente de política")
    print("\n3. Target Policy Smoothing:")
    print("   - Añade ruido a acciones target")
    print("   - Suaviza superficie de Q, más robusto a errores")
    print("\nResultado: TD3 es más estable y performante que DDPG")
    print("=" * 70)


if __name__ == "__main__":
    main()
