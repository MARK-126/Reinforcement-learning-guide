"""
Soft Actor-Critic (SAC)
========================

Implementación de SAC, un algoritmo SOTA para control continuo que combina
maximum entropy RL con actor-critic off-policy.

SAC maximiza tanto recompensas como entropía de la política:
J(π) = E[Σ r_t + α H(π(·|s_t))]

donde H(π) es la entropía y α es el temperature parameter.

Características clave de SAC:

1. **Maximum Entropy Framework**:
   - Política estocástica que maximiza entropía
   - Fomenta exploración y robustez
   - Aprende múltiples soluciones

2. **Automatic Temperature Tuning**:
   - α se ajusta automáticamente
   - Mantiene entropía objetivo
   - No requiere tuning manual

3. **Twin Q-networks**:
   - Dos critics como TD3
   - Reduce sobreestimación

4. **Stochastic Policy**:
   - Gaussian policy con reparameterization trick
   - a = tanh(μ + σ ⊙ ε), ε ~ N(0, I)
   - Permite gradient descent

5. **Off-policy**:
   - Replay buffer para sample efficiency
   - Aprende de experiencias pasadas

Actor objective:
J_π = E[α log π(a|s) - Q(s,a)]

Critic objective:
J_Q = E[(Q(s,a) - (r + γ(min Q'(s',a') - α log π(a'|s'))))²]

Temperature objective:
J_α = E[-α(log π(a|s) + H̄)]

donde H̄ es la entropía objetivo.

Papers:
- "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (Haarnoja et al., 2018)
- "Soft Actor-Critic Algorithms and Applications" (Haarnoja et al., 2019)

Autor: MARK-126
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
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


class GaussianActor(nn.Module):
    """
    Red Actor Gaussiana para SAC: Política estocástica π_θ(a|s).

    Parametriza una distribución Gaussian con:
    - Media: μ_θ(s)
    - Desviación estándar: σ_θ(s)

    Usa reparameterization trick para gradientes:
    a = tanh(μ + σ ⊙ ε), ε ~ N(0, I)

    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    hidden_dims : List[int]
        Dimensiones de capas ocultas
    log_std_min : float
        Límite inferior para log_std
    log_std_max : float
        Límite superior para log_std
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        super(GaussianActor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Capas compartidas
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Heads para mean y log_std
        self.mean = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Linear(prev_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: estado -> (mean, log_std).

        Args:
            state: Estados [batch_size, state_dim]

        Returns:
            mean: Media [batch_size, action_dim]
            log_std: Log desviación estándar [batch_size, action_dim]
        """
        x = self.shared(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Muestrea acción usando reparameterization trick.

        a = tanh(μ + σ ⊙ ε), ε ~ N(0, I)

        Args:
            state: Estados [batch_size, state_dim]

        Returns:
            action: Acciones [batch_size, action_dim] en [-1, 1]
            log_prob: Log probabilidades [batch_size]
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick
        normal = Normal(mean, std)
        z = normal.rsample()  # Reparameterized sample
        action = torch.tanh(z)

        # Calcular log probabilidad con corrección por tanh
        log_prob = normal.log_prob(z)
        # Corrección por squashing (tanh transformation)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Obtiene acción (para evaluación).

        Args:
            state: Estado [batch_size, state_dim]
            deterministic: Si True, usa mean; si False, samplea

        Returns:
            action: Acción [batch_size, action_dim]
        """
        mean, log_std = self.forward(state)

        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)

        return action


class Critic(nn.Module):
    """
    Red Twin Critic: Dos Q-functions Q_φ(s,a).

    Estima valor de pares estado-acción.
    Usa dos redes para reducir sobreestimación.

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
            q1: Q1-values [batch_size, 1]
            q2: Q2-values [batch_size, 1]
        """
        sa = torch.cat([state, action], dim=1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2


class SACAgent:
    """
    Agente Soft Actor-Critic (SAC).

    SAC combina maximum entropy RL con actor-critic off-policy
    para lograr sample efficiency, estabilidad y exploración.

    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    actor_lr : float
        Tasa de aprendizaje del actor
    critic_lr : float
        Tasa de aprendizaje de los critics
    alpha_lr : float
        Tasa de aprendizaje de temperature α
    gamma : float
        Factor de descuento
    tau : float
        Factor para soft update de target networks
    alpha : float
        Temperature inicial (si auto_tune=False)
    auto_tune : bool
        Si True, ajusta α automáticamente
    target_entropy : Optional[float]
        Entropía objetivo (si None, usa -action_dim)
    buffer_size : int
        Tamaño del replay buffer
    batch_size : int
        Tamaño del mini-batch
    hidden_dims : List[int]
        Dimensiones de capas ocultas
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_tune: bool = True,
        target_entropy: Optional[float] = None,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        hidden_dims: List[int] = [256, 256]
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_tune = auto_tune

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Actor
        self.actor = GaussianActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin Critics
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Temperature α (entropy coefficient)
        if auto_tune:
            # Entropía objetivo: -action_dim (heurística común)
            self.target_entropy = target_entropy if target_entropy else -action_dim
            # α como parámetro aprendido
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            print(f"Auto-tuning α activado. Target entropy: {self.target_entropy}")
        else:
            self.alpha = alpha
            print(f"α fijo: {self.alpha}")

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Historial
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_losses': [],
            'alphas': [],
            'q1_values': [],
            'q2_values': [],
            'entropies': []
        }

    @property
    def alpha(self) -> float:
        """Retorna valor actual de α."""
        if self.auto_tune:
            return self.log_alpha.exp().item()
        else:
            return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        """Setter para α (solo si auto_tune=False)."""
        self._alpha = value

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Selecciona acción.

        Args:
            state: Estado actual
            deterministic: Si True, usa mean; si False, samplea

        Returns:
            action: Acción seleccionada [action_dim]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor.get_action(state_tensor, deterministic)

        return action.cpu().numpy().flatten()

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Almacena transición en replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Realiza un paso de entrenamiento SAC.

        Actualiza:
        1. Critics (Q-functions)
        2. Actor (policy)
        3. Temperature α (si auto_tune=True)

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
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        # ========== Actualizar Critics ==========
        with torch.no_grad():
            # Samplear acciones para next_state
            next_action, next_log_prob = self.actor.sample(next_state_batch)

            # Target Q-values con twin critics
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)

            # Target value con entropy bonus
            target_value = target_q - self.alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_value

        # Current Q-values
        current_q1, current_q2 = self.critic(state_batch, action_batch)

        # Critic loss: MSE para ambos critics
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimizar Critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ========== Actualizar Actor ==========
        # Samplear nuevas acciones
        new_action, log_prob = self.actor.sample(state_batch)

        # Q-values para nuevas acciones
        q1_new, q2_new = self.critic(state_batch, new_action)
        q_new = torch.min(q1_new, q2_new)

        # Actor loss: maximizar Q - α*entropy
        # (equivalente a minimizar -Q + α*entropy)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        # Optimizar Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ========== Actualizar Temperature α ==========
        alpha_loss = None
        if self.auto_tune:
            # α loss: minimizar divergencia entre entropía actual y objetivo
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            # Optimizar α
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # ========== Soft update de target critics ==========
        self.soft_update(self.critic, self.critic_target)

        # Retornar métricas
        metrics = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_q1': current_q1.mean().item(),
            'mean_q2': current_q2.mean().item(),
            'alpha': self.alpha,
            'entropy': -log_prob.mean().item()
        }

        if alpha_loss is not None:
            metrics['alpha_loss'] = alpha_loss.item()

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
        warmup_steps: int = 10000,
        updates_per_step: int = 1,
        print_every: int = 10,
        save_every: int = 100,
        save_path: str = "sac_model.pth"
    ) -> Dict[str, List]:
        """
        Entrena el agente SAC.

        Args:
            env: Entorno gymnasium
            n_episodes: Número de episodios
            max_steps: Pasos máximos por episodio
            warmup_steps: Steps de exploración aleatoria inicial
            updates_per_step: Actualizaciones por step del ambiente
            print_every: Frecuencia de logging
            save_every: Frecuencia de guardado
            save_path: Ruta para guardar modelo

        Returns:
            history: Historial de entrenamiento
        """
        print(f"Entrenando SAC...")
        print(f"Episodios: {n_episodes}, Warmup: {warmup_steps}")
        print(f"Gamma: {self.gamma}, Tau: {self.tau}")
        print(f"Updates per step: {updates_per_step}\n")

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
                    action = self.get_action(state, deterministic=False)

                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Almacenar transición
                self.store_transition(state, action, reward, next_state, done)

                # Entrenar múltiples veces por step (después de warmup)
                if total_steps >= warmup_steps:
                    for _ in range(updates_per_step):
                        metrics = self.train_step()
                        if metrics is not None:
                            episode_metrics.append(metrics)

                episode_reward += reward
                total_steps += 1
                state = next_state

                if done:
                    break

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
                self.history['q1_values'].append(avg_metrics['mean_q1'])
                self.history['q2_values'].append(avg_metrics['mean_q2'])
                self.history['alphas'].append(avg_metrics['alpha'])
                self.history['entropies'].append(avg_metrics['entropy'])

                if 'alpha_loss' in avg_metrics:
                    self.history['alpha_losses'].append(avg_metrics['alpha_loss'])

            # Logging
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.history['episode_rewards'][-100:])
                avg_length = np.mean(self.history['episode_lengths'][-100:])
                avg_q1 = np.mean(self.history['q1_values'][-100:]) if self.history['q1_values'] else 0
                avg_entropy = np.mean(self.history['entropies'][-100:]) if self.history['entropies'] else 0

                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg(100): {avg_reward:.2f} | "
                      f"Length: {step + 1} | "
                      f"α: {self.alpha:.3f} | "
                      f"Q1: {avg_q1:.2f} | "
                      f"Entropy: {avg_entropy:.3f}")

            # Guardar modelo
            if save_every > 0 and (episode + 1) % save_every == 0:
                self.save(save_path)

        print("\nEntrenamiento completado!")
        return self.history

    def save(self, path: str):
        """Guarda el modelo."""
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'history': self.history
        }

        if self.auto_tune:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        else:
            save_dict['alpha'] = self.alpha

        torch.save(save_dict, path)
        print(f"Modelo guardado en {path}")

    def load(self, path: str):
        """Carga el modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.history = checkpoint['history']

        if self.auto_tune:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        else:
            self.alpha = checkpoint['alpha']

        print(f"Modelo cargado desde {path}")


def evaluate_agent(agent: SACAgent, env: gym.Env,
                   n_episodes: int = 10, render: bool = False) -> Tuple[float, float]:
    """
    Evalúa el agente entrenado.

    Args:
        agent: Agente SAC
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
            action = agent.get_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if render:
                env.render()

        rewards.append(episode_reward)

    return np.mean(rewards), np.std(rewards)


def plot_training_results(history: Dict[str, List], save_path: str = 'sac_training.png'):
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

    # Temperature α
    ax = axes[0, 2]
    if history['alphas']:
        ax.plot(history['alphas'], alpha=0.6, label='α (temperature)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('α')
    ax.set_title('Temperature α (Entropy Coefficient)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropía
    ax = axes[1, 0]
    if history['entropies']:
        ax.plot(history['entropies'], alpha=0.6, label='Policy Entropy')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-values
    ax = axes[1, 1]
    if history['q1_values'] and history['q2_values']:
        ax.plot(history['q1_values'], alpha=0.6, label='Q1')
        ax.plot(history['q2_values'], alpha=0.6, label='Q2')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q-value')
    ax.set_title('Twin Q-values')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode Lengths
    ax = axes[1, 2]
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
    """Ejemplos de uso: Entrenar SAC en ambientes continuos"""

    print("=" * 70)
    print("SOFT ACTOR-CRITIC (SAC)")
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

    agent_pendulum = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_tune=True,
        target_entropy=None,  # -action_dim
        buffer_size=1000000,
        batch_size=256,
        hidden_dims=[256, 256]
    )

    history_pendulum = agent_pendulum.train(
        env=env_pendulum,
        n_episodes=150,
        max_steps=200,
        warmup_steps=1000,
        updates_per_step=1,
        print_every=10,
        save_every=50,
        save_path='sac_pendulum.pth'
    )

    plot_training_results(history_pendulum, 'sac_pendulum_training.png')

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

    agent_mountain = SACAgent(
        state_dim=state_dim_m,
        action_dim=action_dim_m,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        auto_tune=True,
        buffer_size=1000000,
        batch_size=256,
        hidden_dims=[256, 256]
    )

    history_mountain = agent_mountain.train(
        env=env_mountain,
        n_episodes=400,
        max_steps=999,
        warmup_steps=5000,
        updates_per_step=1,
        print_every=20,
        save_every=100,
        save_path='sac_mountain.pth'
    )

    plot_training_results(history_mountain, 'sac_mountain_training.png')

    print("\nEvaluando política aprendida...")
    mean_reward_m, std_reward_m = evaluate_agent(agent_mountain, env_mountain, n_episodes=30)
    print(f"Recompensa promedio: {mean_reward_m:.2f} ± {std_reward_m:.2f}")

    env_mountain.close()

    print("\n" + "=" * 70)
    print("VENTAJAS DE SAC")
    print("=" * 70)
    print("1. Maximum Entropy Framework:")
    print("   - Fomenta exploración naturalmente")
    print("   - Aprende políticas robustas y multimodales")
    print("\n2. Automatic Temperature Tuning:")
    print("   - α se ajusta automáticamente")
    print("   - No requiere tuning manual")
    print("\n3. Sample Efficiency:")
    print("   - Off-policy con replay buffer")
    print("   - Reutiliza experiencias pasadas")
    print("\n4. Stability:")
    print("   - Twin Q-networks reducen sobreestimación")
    print("   - Stochastic policy más robusta que deterministic")
    print("\n5. State-of-the-Art:")
    print("   - Uno de los mejores algoritmos para control continuo")
    print("   - Usado en robótica real, juegos, etc.")
    print("=" * 70)


if __name__ == "__main__":
    main()
