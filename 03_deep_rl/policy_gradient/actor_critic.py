"""
Advantage Actor-Critic (A2C)
==============================

Implementación del algoritmo Advantage Actor-Critic, que combina Policy Gradient
con métodos de Temporal Difference para reducir varianza y mejorar eficiencia.

A2C mantiene dos redes:
1. Actor: Política π_θ(a|s) que selecciona acciones
2. Critic: Value function V_φ(s) que estima el valor de estados

El actor se actualiza usando el gradiente de política con ventaja:
∇J(θ) = E[∇log π_θ(a|s) * A(s,a)]

donde A(s,a) = Q(s,a) - V(s) es la ventaja.

El critic se actualiza minimizando el error TD:
L(φ) = E[(r + γV(s') - V(s))²]

Características:
- On-policy: Aprende de su propia experiencia
- Temporal Difference: Más eficiente que Monte Carlo
- Menor varianza que REINFORCE
- Soporta GAE (Generalized Advantage Estimation)
- Regularización de entropía para exploración

Variantes implementadas:
1. A2C básico con TD(0)
2. A2C con n-step returns
3. A2C con GAE (λ-returns)

Papers:
- "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)
- "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (Schulman et al., 2016)

Autor: MARK-126
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import gymnasium as gym
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import matplotlib.pyplot as plt


class ActorNetwork(nn.Module):
    """
    Red Actor: Política π_θ(a|s).

    Salida:
    - Discreto: Logits sobre acciones
    - Continuo: Media y log_std de distribución Normal

    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    hidden_dims : List[int]
        Dimensiones de capas ocultas
    continuous : bool
        True para acciones continuas
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 continuous: bool = False):
        super(ActorNetwork, self).__init__()

        self.continuous = continuous
        self.action_dim = action_dim

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

        if continuous:
            self.mean = nn.Linear(prev_dim, action_dim)
            self.log_std = nn.Linear(prev_dim, action_dim)
        else:
            self.action_head = nn.Linear(prev_dim, action_dim)

    def forward(self, state: torch.Tensor):
        """Forward pass."""
        x = self.shared(state)

        if self.continuous:
            mean = self.mean(x)
            log_std = self.log_std(x)
            log_std = torch.clamp(log_std, -20, 2)
            return mean, log_std
        else:
            logits = self.action_head(x)
            return logits

    def get_action_and_log_prob(self, state: torch.Tensor, action: Optional[torch.Tensor] = None):
        """
        Obtiene acción y log probabilidad.

        Args:
            state: Estado [batch_size, state_dim]
            action: Acción (si se provee, calcula su log_prob; si no, samplea)

        Returns:
            action: Acción seleccionada
            log_prob: Log probabilidad
            entropy: Entropía de la distribución
        """
        if self.continuous:
            mean, log_std = self.forward(state)
            std = log_std.exp()
            dist = Normal(mean, std)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = self.forward(state)
            dist = Categorical(logits=logits)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Red Critic: Función de valor V_φ(s).

    Estima el valor esperado de un estado bajo la política actual.

    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    hidden_dims : List[int]
        Dimensiones de capas ocultas
    """

    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 256]):
        super(CriticNetwork, self).__init__()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Estados [batch_size, state_dim]

        Returns:
            values: Valores [batch_size]
        """
        return self.network(state).squeeze(-1)


class A2CAgent:
    """
    Agente Advantage Actor-Critic (A2C).

    Implementa A2C con soporte para:
    - Acciones discretas y continuas
    - GAE (Generalized Advantage Estimation)
    - n-step returns
    - Regularización de entropía
    - Normalización de ventajas

    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool
        True para acciones continuas
    actor_lr : float
        Tasa de aprendizaje del actor
    critic_lr : float
        Tasa de aprendizaje del critic
    gamma : float
        Factor de descuento
    gae_lambda : float
        Parámetro λ para GAE (0 = TD(0), 1 = Monte Carlo)
    entropy_coef : float
        Coeficiente de regularización de entropía
    value_loss_coef : float
        Coeficiente de loss del value function
    max_grad_norm : float
        Límite para gradient clipping
    normalize_advantages : bool
        Si True, normaliza ventajas
    n_steps : int
        Número de steps para n-step returns (si None, usa episodios completos)
    use_gae : bool
        Si True, usa GAE; si False, usa n-step returns simples
    hidden_dims : List[int]
        Dimensiones de capas ocultas
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        continuous: bool = False,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
        n_steps: Optional[int] = None,
        use_gae: bool = True,
        hidden_dims: List[int] = [256, 256]
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        self.n_steps = n_steps
        self.use_gae = use_gae

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Redes
        self.actor = ActorNetwork(
            state_dim, action_dim, hidden_dims, continuous
        ).to(self.device)

        self.critic = CriticNetwork(state_dim, hidden_dims).to(self.device)

        # Optimizadores
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Historial
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropies': [],
            'advantages': []
        }

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Selecciona acción para un estado.

        Args:
            state: Estado actual
            deterministic: Si True, acción determinista

        Returns:
            action: Acción seleccionada
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                if self.continuous:
                    mean, _ = self.actor.forward(state_tensor)
                    action = mean
                else:
                    logits = self.actor.forward(state_tensor)
                    action = logits.argmax(dim=-1)
            else:
                action, _, _ = self.actor.get_action_and_log_prob(state_tensor)

        if self.continuous:
            return action.cpu().numpy().flatten()
        else:
            return action.item()

    def compute_gae(
        self,
        rewards: List[float],
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula Generalized Advantage Estimation (GAE).

        GAE(λ) combina n-step returns con diferentes n:
        A^GAE_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

        donde δ_t = r_t + γV(s_{t+1}) - V(s_t) es el TD error.

        Args:
            rewards: Lista de recompensas
            values: Valores V(s_t)
            next_values: Valores V(s_{t+1})
            dones: Lista de flags done

        Returns:
            advantages: Ventajas calculadas
            returns: Retornos (targets para value function)
        """
        advantages = []
        gae = 0

        # Calcular GAE en reversa
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            # TD error: δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

            # GAE: A_t = δ_t + γλ * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + values

        return advantages, returns

    def compute_n_step_returns(
        self,
        rewards: List[float],
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula n-step returns.

        R_t^(n) = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})

        Args:
            rewards: Lista de recompensas
            values: Valores V(s_t)
            next_values: Valores V(s_{t+1})
            dones: Lista de flags done

        Returns:
            advantages: Ventajas (returns - values)
            returns: Retornos
        """
        returns = []
        R = next_values[-1] if len(rewards) > 0 else 0

        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R * (1 - dones[t])
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - values

        return advantages, returns

    def train_step(
        self,
        states: List[np.ndarray],
        actions: List,
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool]
    ) -> Dict[str, float]:
        """
        Realiza un paso de entrenamiento (puede ser episodio completo o n-steps).

        Args:
            states: Lista de estados
            actions: Lista de acciones
            rewards: Lista de recompensas
            next_states: Lista de siguientes estados
            dones: Lista de flags done

        Returns:
            metrics: Diccionario con métricas de entrenamiento
        """
        # Convertir a tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)

        if self.continuous:
            actions_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        else:
            actions_tensor = torch.LongTensor(actions).to(self.device)

        # Calcular valores
        with torch.no_grad():
            values = self.critic(states_tensor)
            next_values = self.critic(next_states_tensor)

        # Calcular ventajas
        if self.use_gae:
            advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        else:
            advantages, returns = self.compute_n_step_returns(rewards, values, next_values, dones)

        # Normalizar ventajas
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass Actor
        _, log_probs, entropies = self.actor.get_action_and_log_prob(
            states_tensor, actions_tensor
        )

        # Actor loss: -E[log π(a|s) * A(s,a)] - entropy_bonus
        actor_loss = -(log_probs * advantages.detach()).mean()
        entropy_loss = -entropies.mean()
        total_actor_loss = actor_loss + self.entropy_coef * entropy_loss

        # Critic loss: MSE entre V(s) y returns
        values_new = self.critic(states_tensor)
        critic_loss = F.mse_loss(values_new, returns.detach())

        # Optimizar Actor
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Optimizar Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropies.mean().item(),
            'mean_advantage': advantages.mean().item(),
            'mean_value': values.mean().item()
        }

    def train(
        self,
        env: gym.Env,
        n_episodes: int = 1000,
        max_steps: int = 500,
        print_every: int = 100,
        render: bool = False
    ) -> Dict[str, List]:
        """
        Entrena el agente A2C.

        Si n_steps está definido, actualiza cada n_steps.
        Si no, actualiza al final de cada episodio.

        Args:
            env: Entorno gymnasium
            n_episodes: Número de episodios
            max_steps: Pasos máximos por episodio
            print_every: Frecuencia de logging
            render: Si True, renderiza

        Returns:
            history: Historial de entrenamiento
        """
        print(f"Entrenando A2C {'con GAE' if self.use_gae else 'con n-step returns'}...")
        print(f"Episodios: {n_episodes}, Gamma: {self.gamma}, "
              f"GAE Lambda: {self.gae_lambda}, Entropy coef: {self.entropy_coef}\n")

        for episode in range(n_episodes):
            state, _ = env.reset()

            # Buffers para el episodio
            states_buffer = []
            actions_buffer = []
            rewards_buffer = []
            next_states_buffer = []
            dones_buffer = []

            episode_reward = 0
            episode_metrics = []

            for step in range(max_steps):
                # Seleccionar acción
                action = self.get_action(state, deterministic=False)

                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Almacenar transición
                states_buffer.append(state)
                actions_buffer.append(action)
                rewards_buffer.append(reward)
                next_states_buffer.append(next_state)
                dones_buffer.append(float(done))

                episode_reward += reward
                state = next_state

                # Actualizar cada n_steps o al final del episodio
                should_update = (
                    (self.n_steps is not None and len(states_buffer) >= self.n_steps) or
                    done
                )

                if should_update and len(states_buffer) > 0:
                    metrics = self.train_step(
                        states_buffer, actions_buffer, rewards_buffer,
                        next_states_buffer, dones_buffer
                    )
                    episode_metrics.append(metrics)

                    # Limpiar buffers si usamos n-steps
                    if self.n_steps is not None:
                        states_buffer = []
                        actions_buffer = []
                        rewards_buffer = []
                        next_states_buffer = []
                        dones_buffer = []

                if done:
                    break

            # Registrar estadísticas del episodio
            self.history['episode_rewards'].append(episode_reward)
            self.history['episode_lengths'].append(step + 1)

            if episode_metrics:
                avg_metrics = {
                    key: np.mean([m[key] for m in episode_metrics])
                    for key in episode_metrics[0].keys()
                }
                self.history['actor_losses'].append(avg_metrics['actor_loss'])
                self.history['critic_losses'].append(avg_metrics['critic_loss'])
                self.history['entropies'].append(avg_metrics['entropy'])
                self.history['advantages'].append(avg_metrics['mean_advantage'])

            # Logging
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.history['episode_rewards'][-100:])
                avg_length = np.mean(self.history['episode_lengths'][-100:])
                avg_entropy = np.mean(self.history['entropies'][-100:]) if self.history['entropies'] else 0

                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg Reward (100): {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Entropy: {avg_entropy:.3f}")

        print("\nEntrenamiento completado!")
        return self.history

    def save(self, path: str):
        """Guarda el modelo."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"Modelo guardado en {path}")

    def load(self, path: str):
        """Carga el modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.history = checkpoint['history']
        print(f"Modelo cargado desde {path}")


def plot_training_results(history: Dict[str, List], save_path: str = 'a2c_training.png'):
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

    # Longitud de episodios
    ax = axes[0, 1]
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

    # Actor Loss
    ax = axes[0, 2]
    if history['actor_losses']:
        losses = history['actor_losses']
        ax.plot(losses, alpha=0.3, label='Actor Loss')
        if len(losses) >= 10:
            window = min(100, len(losses) // 10)
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(losses)), moving_avg,
                   label=f'MA({window})', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Actor Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Critic Loss
    ax = axes[1, 0]
    if history['critic_losses']:
        losses = history['critic_losses']
        ax.plot(losses, alpha=0.3, label='Critic Loss')
        if len(losses) >= 10:
            window = min(100, len(losses) // 10)
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(losses)), moving_avg,
                   label=f'MA({window})', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Critic Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropía
    ax = axes[1, 1]
    if history['entropies']:
        entropies = history['entropies']
        ax.plot(entropies, alpha=0.3, label='Entropy')
        if len(entropies) >= 10:
            window = min(100, len(entropies) // 10)
            moving_avg = np.convolve(entropies, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(entropies)), moving_avg,
                   label=f'MA({window})', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ventajas
    ax = axes[1, 2]
    if history['advantages']:
        advantages = history['advantages']
        ax.plot(advantages, alpha=0.3, label='Mean Advantage')
        if len(advantages) >= 10:
            window = min(100, len(advantages) // 10)
            moving_avg = np.convolve(advantages, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(advantages)), moving_avg,
                   label=f'MA({window})', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Advantage')
    ax.set_title('Mean Advantage')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráficos guardados en '{save_path}'")


def evaluate_agent(agent: A2CAgent, env: gym.Env,
                   n_episodes: int = 10, render: bool = False) -> Tuple[float, float]:
    """
    Evalúa el agente entrenado.

    Args:
        agent: Agente A2C
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


# ==================== MAIN ====================

def main():
    """Ejemplos de uso: Entrenar A2C en CartPole y Pendulum"""

    print("=" * 70)
    print("ADVANTAGE ACTOR-CRITIC (A2C)")
    print("=" * 70)

    # ========== Ejemplo 1: CartPole (Discreto) ==========
    print("\n" + "=" * 70)
    print("Ejemplo 1: CartPole-v1 (Acciones Discretas)")
    print("=" * 70)

    env_cartpole = gym.make('CartPole-v1')
    state_dim = env_cartpole.observation_space.shape[0]
    action_dim = env_cartpole.action_space.n

    print(f"\nEntorno: CartPole-v1")
    print(f"Estado: {state_dim} dimensiones")
    print(f"Acciones: {action_dim} (discretas)\n")

    # A2C con GAE
    agent_gae = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=False,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantages=True,
        n_steps=None,  # Actualizar por episodio completo
        use_gae=True,
        hidden_dims=[256, 256]
    )

    print("Entrenando A2C con GAE...")
    history_gae = agent_gae.train(
        env=env_cartpole,
        n_episodes=500,
        max_steps=500,
        print_every=50
    )

    # Visualizar
    plot_training_results(history_gae, 'a2c_cartpole_gae.png')

    # Evaluar
    print("\nEvaluando política aprendida...")
    mean_reward, std_reward = evaluate_agent(agent_gae, env_cartpole, n_episodes=100)
    print(f"Recompensa promedio: {mean_reward:.2f} ± {std_reward:.2f}")

    # Guardar modelo
    agent_gae.save('a2c_cartpole.pth')

    env_cartpole.close()

    # ========== Ejemplo 2: Pendulum (Continuo) ==========
    print("\n" + "=" * 70)
    print("Ejemplo 2: Pendulum-v1 (Acciones Continuas)")
    print("=" * 70)

    env_pendulum = gym.make('Pendulum-v1')
    state_dim_p = env_pendulum.observation_space.shape[0]
    action_dim_p = env_pendulum.action_space.shape[0]

    print(f"\nEntorno: Pendulum-v1")
    print(f"Estado: {state_dim_p} dimensiones")
    print(f"Acciones: {action_dim_p} (continuas)\n")

    # A2C para continuo con n-steps
    agent_continuous = A2CAgent(
        state_dim=state_dim_p,
        action_dim=action_dim_p,
        continuous=True,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.001,  # Menos entropía para continuo
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantages=True,
        n_steps=5,  # Actualizar cada 5 steps
        use_gae=True,
        hidden_dims=[256, 256]
    )

    print("Entrenando A2C (acciones continuas) con n-steps y GAE...")
    history_continuous = agent_continuous.train(
        env=env_pendulum,
        n_episodes=300,
        max_steps=200,
        print_every=30
    )

    # Visualizar
    plot_training_results(history_continuous, 'a2c_pendulum_continuous.png')

    # Evaluar
    print("\nEvaluando política aprendida...")
    mean_reward_p, std_reward_p = evaluate_agent(
        agent_continuous, env_pendulum, n_episodes=50
    )
    print(f"Recompensa promedio: {mean_reward_p:.2f} ± {std_reward_p:.2f}")

    # Guardar modelo
    agent_continuous.save('a2c_pendulum.pth')

    env_pendulum.close()

    # ========== Comparación: GAE vs n-step simple ==========
    print("\n" + "=" * 70)
    print("Ejemplo 3: Comparación GAE vs n-step simple")
    print("=" * 70)

    env_compare = gym.make('CartPole-v1')

    # A2C sin GAE
    agent_no_gae = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=False,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        normalize_advantages=True,
        n_steps=None,
        use_gae=False,  # Sin GAE
        hidden_dims=[256, 256]
    )

    print("\nEntrenando A2C sin GAE...")
    history_no_gae = agent_no_gae.train(
        env=env_compare,
        n_episodes=500,
        max_steps=500,
        print_every=50
    )

    mean_reward_no_gae, std_reward_no_gae = evaluate_agent(
        agent_no_gae, env_compare, n_episodes=100
    )

    print("\n" + "=" * 70)
    print("COMPARACIÓN DE RESULTADOS")
    print("=" * 70)
    print(f"A2C con GAE:     {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"A2C sin GAE:     {mean_reward_no_gae:.2f} ± {std_reward_no_gae:.2f}")
    print("\nConclusión: GAE generalmente reduce varianza y mejora estabilidad.")
    print("=" * 70)

    env_compare.close()


if __name__ == "__main__":
    main()
