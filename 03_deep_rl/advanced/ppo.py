"""
Proximal Policy Optimization (PPO)
====================================

Implementación de PPO, uno de los algoritmos de Deep RL más populares y efectivos.
PPO mejora sobre TRPO usando un objetivo clipped más simple pero igual de efectivo.

PPO resuelve el problema de:
1. Mantener updates de política seguros (no demasiado grandes)
2. Ser computacionalmente eficiente (sin KL constraint complicado)
3. Funcionar bien en diversos ambientes

Características clave:
- Clipped surrogate objective: limita cambios de política
- GAE (Generalized Advantage Estimation)
- Mini-batch training con múltiples epochs
- Value function clipping
- Entropy bonus para exploración
- Soporte para acciones discretas y continuas

Objetivo PPO-Clip:
L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

donde r_t(θ) = π_θ(a|s) / π_θ_old(a|s) es el probability ratio.

Papers:
- "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- "High-Dimensional Continuous Control Using GAE" (Schulman et al., 2016)

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
    Red Actor para la política π_θ(a|s).

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
        True para acciones continuas, False para discretas
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [64, 64],
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
                nn.Tanh()
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        if continuous:
            # Para acciones continuas: salida es (mean, log_std)
            self.mean = nn.Linear(prev_dim, action_dim)
            # Log_std puede ser parámetro aprendido o por estado
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # Para acciones discretas: salida es logits
            self.action_head = nn.Linear(prev_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            state: Estados [batch_size, state_dim]

        Returns:
            - Discreto: (logits, None)
            - Continuo: (mean, log_std)
        """
        x = self.shared(state)

        if self.continuous:
            mean = self.mean(x)
            log_std = self.log_std.expand_as(mean)
            return mean, log_std
        else:
            logits = self.action_head(x)
            return logits, None

    def get_action_and_log_prob(self, state: torch.Tensor,
                                action: Optional[torch.Tensor] = None):
        """
        Obtiene acción y log probabilidad.

        Args:
            state: Estado [batch_size, state_dim]
            action: Acción (si None, samplea nueva acción)

        Returns:
            action: Acción seleccionada
            log_prob: Log probabilidad de la acción
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
            logits, _ = self.forward(state)
            dist = Categorical(logits=logits)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Red Critic para la función de valor V(s).

    Estima el valor esperado de un estado.

    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    hidden_dims : List[int]
        Dimensiones de capas ocultas
    """

    def __init__(self, state_dim: int, hidden_dims: List[int] = [64, 64]):
        super(CriticNetwork, self).__init__()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh()
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


class PPOAgent:
    """
    Agente Proximal Policy Optimization (PPO).

    PPO es un algoritmo on-policy que usa clipping para asegurar
    updates seguros de la política. Es uno de los métodos SOTA
    más confiables y utilizados en Deep RL.

    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool
        True para acciones continuas
    lr : float
        Tasa de aprendizaje
    gamma : float
        Factor de descuento
    gae_lambda : float
        Parámetro λ para GAE
    epsilon_clip : float
        Epsilon para clipping del ratio de probabilidad
    value_clip : float
        Epsilon para clipping del value loss (None = sin clipping)
    entropy_coef : float
        Coeficiente de bonus de entropía
    value_loss_coef : float
        Coeficiente de loss del value function
    max_grad_norm : float
        Límite para gradient clipping
    n_epochs : int
        Número de epochs de optimización por batch
    batch_size : int
        Tamaño de mini-batch
    normalize_advantages : bool
        Si True, normaliza ventajas
    hidden_dims : List[int]
        Dimensiones de capas ocultas
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        continuous: bool = False,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epsilon_clip: float = 0.2,
        value_clip: Optional[float] = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        normalize_advantages: bool = True,
        hidden_dims: List[int] = [64, 64]
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon_clip = epsilon_clip
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.normalize_advantages = normalize_advantages

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Redes
        self.actor = ActorNetwork(
            state_dim, action_dim, hidden_dims, continuous
        ).to(self.device)

        self.critic = CriticNetwork(state_dim, hidden_dims).to(self.device)

        # Optimizador único para ambas redes (como en paper original)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        # Historial
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'total_losses': [],
            'entropies': [],
            'kl_divs': [],
            'clip_fractions': []
        }

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Selecciona acción para un estado.

        Args:
            state: Estado actual
            deterministic: Si True, acción determinista

        Returns:
            action: Acción seleccionada (numpy array)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                if self.continuous:
                    mean, _ = self.actor.forward(state_tensor)
                    action = mean
                else:
                    logits, _ = self.actor.forward(state_tensor)
                    action = logits.argmax(dim=-1)
            else:
                action, _, _ = self.actor.get_action_and_log_prob(state_tensor)

        if self.continuous:
            return action.cpu().numpy().flatten()
        else:
            return action.item()

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: torch.Tensor,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula Generalized Advantage Estimation (GAE).

        GAE reduce varianza de estimación de ventajas mediante
        promedio exponencial de n-step advantages.

        Args:
            rewards: Recompensas [T]
            values: Valores V(s) [T]
            dones: Flags done [T]
            next_value: Valor del último next_state

        Returns:
            advantages: Ventajas [T]
            returns: Retornos (targets para value) [T]
        """
        advantages = []
        gae = 0

        values = values.cpu().numpy()
        next_values = np.append(values[1:], next_value)

        # Calcular GAE en reversa
        for t in reversed(range(len(rewards))):
            # TD error: δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]

            # GAE: A_t = δ_t + γλ(1-done) * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)

        return advantages, returns

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor
    ) -> Dict[str, float]:
        """
        Actualiza actor y critic usando PPO.

        Realiza múltiples epochs de optimización sobre mini-batches
        del buffer recolectado.

        Args:
            states: Estados [N, state_dim]
            actions: Acciones [N, action_dim] o [N]
            old_log_probs: Log probs bajo política antigua [N]
            advantages: Ventajas [N]
            returns: Retornos [N]
            old_values: Valores bajo value function antigua [N]

        Returns:
            metrics: Diccionario con métricas de entrenamiento
        """
        # Convertir a tensors
        states = torch.FloatTensor(states).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

        if self.continuous:
            actions = torch.FloatTensor(actions).to(self.device)
        else:
            actions = torch.LongTensor(actions).to(self.device)

        # Normalizar ventajas
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Métricas acumuladas
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_kl = 0
        total_clip_fraction = 0
        n_updates = 0

        # Múltiples epochs de optimización
        for epoch in range(self.n_epochs):
            # Crear mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]

                # Evaluar acciones bajo política actual
                _, new_log_probs, entropy = self.actor.get_action_and_log_prob(
                    batch_states, batch_actions
                )

                # Ratio de probabilidad: r(θ) = π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO Clipped Surrogate Objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                new_values = self.critic(batch_states)

                if self.value_clip is not None:
                    # Value function clipping (ayuda a estabilidad)
                    value_pred_clipped = batch_old_values + torch.clamp(
                        new_values - batch_old_values,
                        -self.value_clip,
                        self.value_clip
                    )
                    value_loss1 = F.mse_loss(new_values, batch_returns)
                    value_loss2 = F.mse_loss(value_pred_clipped, batch_returns)
                    critic_loss = torch.max(value_loss1, value_loss2)
                else:
                    critic_loss = F.mse_loss(new_values, batch_returns)

                # Entropy bonus (fomenta exploración)
                entropy_loss = -entropy.mean()

                # Loss total
                loss = (actor_loss +
                       self.value_loss_coef * critic_loss +
                       self.entropy_coef * entropy_loss)

                # Optimización
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # Métricas
                with torch.no_grad():
                    # KL divergence aproximado
                    kl = (batch_old_log_probs - new_log_probs).mean()
                    # Fracción de ratios que fueron clipped
                    clip_fraction = ((ratio - 1.0).abs() > self.epsilon_clip).float().mean()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += kl.item()
                total_clip_fraction += clip_fraction.item()
                n_updates += 1

        # Promediar métricas
        return {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'total_loss': (total_actor_loss + total_critic_loss) / n_updates,
            'entropy': total_entropy / n_updates,
            'kl_div': total_kl / n_updates,
            'clip_fraction': total_clip_fraction / n_updates
        }

    def train(
        self,
        env: gym.Env,
        n_episodes: int = 1000,
        max_steps: int = 1000,
        update_interval: int = 2048,
        print_every: int = 10,
        save_every: int = 100,
        save_path: str = "ppo_model.pth"
    ) -> Dict[str, List]:
        """
        Entrena el agente PPO.

        Recolecta experiencias durante update_interval steps,
        luego actualiza la política con múltiples epochs.

        Args:
            env: Entorno gymnasium
            n_episodes: Número de episodios
            max_steps: Pasos máximos por episodio
            update_interval: Steps antes de actualizar política
            print_every: Frecuencia de logging (episodios)
            save_every: Frecuencia de guardado (episodios)
            save_path: Ruta para guardar modelo

        Returns:
            history: Historial de entrenamiento
        """
        print(f"Entrenando PPO...")
        print(f"Episodios: {n_episodes}, Update interval: {update_interval}")
        print(f"Epsilon clip: {self.epsilon_clip}, Value clip: {self.value_clip}")
        print(f"GAE lambda: {self.gae_lambda}, Epochs: {self.n_epochs}\n")

        # Buffers para recolección
        states_buffer = []
        actions_buffer = []
        rewards_buffer = []
        dones_buffer = []
        log_probs_buffer = []
        values_buffer = []

        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        total_steps = 0

        for episode in range(n_episodes):
            episode_done = False

            while not episode_done:
                # Recolectar experiencias
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action, log_prob, _ = self.actor.get_action_and_log_prob(state_tensor)
                    value = self.critic(state_tensor)

                # Ejecutar acción
                if self.continuous:
                    action_env = action.cpu().numpy().flatten()
                else:
                    action_env = action.item()

                next_state, reward, terminated, truncated, _ = env.step(action_env)
                done = terminated or truncated

                # Almacenar transición
                states_buffer.append(state)
                actions_buffer.append(action_env)
                rewards_buffer.append(reward)
                dones_buffer.append(float(done))
                log_probs_buffer.append(log_prob.item())
                values_buffer.append(value.item())

                episode_reward += reward
                episode_length += 1
                total_steps += 1
                state = next_state

                # Actualizar política cada update_interval steps
                if total_steps % update_interval == 0 or done:
                    if len(states_buffer) > 0:
                        # Calcular next_value para GAE
                        with torch.no_grad():
                            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                            next_value = self.critic(next_state_tensor).item() if not done else 0.0

                        # Convertir buffers a arrays
                        states_array = np.array(states_buffer)
                        actions_array = np.array(actions_buffer)
                        rewards_array = np.array(rewards_buffer)
                        dones_array = np.array(dones_buffer)
                        log_probs_array = np.array(log_probs_buffer)
                        values_tensor = torch.FloatTensor(values_buffer).to(self.device)

                        # Calcular GAE
                        advantages, returns = self.compute_gae(
                            rewards_array, values_tensor, dones_array, next_value
                        )

                        # Actualizar política
                        metrics = self.update(
                            states_array, actions_array, log_probs_array,
                            advantages, returns, values_tensor
                        )

                        # Registrar métricas
                        self.history['actor_losses'].append(metrics['actor_loss'])
                        self.history['critic_losses'].append(metrics['critic_loss'])
                        self.history['total_losses'].append(metrics['total_loss'])
                        self.history['entropies'].append(metrics['entropy'])
                        self.history['kl_divs'].append(metrics['kl_div'])
                        self.history['clip_fractions'].append(metrics['clip_fraction'])

                        # Limpiar buffers
                        states_buffer = []
                        actions_buffer = []
                        rewards_buffer = []
                        dones_buffer = []
                        log_probs_buffer = []
                        values_buffer = []

                if done:
                    # Registrar episodio
                    self.history['episode_rewards'].append(episode_reward)
                    self.history['episode_lengths'].append(episode_length)

                    # Logging
                    if (episode + 1) % print_every == 0:
                        avg_reward = np.mean(self.history['episode_rewards'][-100:])
                        avg_length = np.mean(self.history['episode_lengths'][-100:])
                        avg_kl = np.mean(self.history['kl_divs'][-10:]) if self.history['kl_divs'] else 0
                        avg_clip = np.mean(self.history['clip_fractions'][-10:]) if self.history['clip_fractions'] else 0

                        print(f"Episode {episode + 1}/{n_episodes} | "
                              f"Reward: {episode_reward:.2f} | "
                              f"Avg(100): {avg_reward:.2f} | "
                              f"Length: {episode_length} | "
                              f"KL: {avg_kl:.4f} | "
                              f"Clip: {avg_clip:.3f}")

                    # Guardar modelo
                    if save_every > 0 and (episode + 1) % save_every == 0:
                        self.save(save_path)

                    # Reset para nuevo episodio
                    state, _ = env.reset()
                    episode_reward = 0
                    episode_length = 0
                    episode_done = True

        print("\nEntrenamiento completado!")
        return self.history

    def save(self, path: str):
        """Guarda el modelo."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"Modelo guardado en {path}")

    def load(self, path: str):
        """Carga el modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.history = checkpoint['history']
        print(f"Modelo cargado desde {path}")


def evaluate_agent(agent: PPOAgent, env: gym.Env,
                   n_episodes: int = 10, render: bool = False) -> Tuple[float, float]:
    """
    Evalúa el agente entrenado.

    Args:
        agent: Agente PPO
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


def plot_training_results(history: Dict[str, List], save_path: str = 'ppo_training.png'):
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
    if history['actor_losses']:
        ax.plot(history['actor_losses'], alpha=0.6, label='Actor Loss')
        ax.plot(history['critic_losses'], alpha=0.6, label='Critic Loss')
    ax.set_xlabel('Update')
    ax.set_ylabel('Loss')
    ax.set_title('Actor & Critic Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropía
    ax = axes[0, 2]
    if history['entropies']:
        ax.plot(history['entropies'], alpha=0.6, label='Entropy')
    ax.set_xlabel('Update')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # KL Divergence
    ax = axes[1, 0]
    if history['kl_divs']:
        ax.plot(history['kl_divs'], alpha=0.6, label='KL Divergence')
    ax.set_xlabel('Update')
    ax.set_ylabel('KL')
    ax.set_title('KL Divergence (old vs new policy)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Clip Fraction
    ax = axes[1, 1]
    if history['clip_fractions']:
        ax.plot(history['clip_fractions'], alpha=0.6, label='Clip Fraction')
        ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Target (20%)')
    ax.set_xlabel('Update')
    ax.set_ylabel('Fraction')
    ax.set_title('Clipping Fraction')
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
    """Ejemplos de uso: Entrenar PPO en diferentes ambientes"""

    print("=" * 70)
    print("PROXIMAL POLICY OPTIMIZATION (PPO)")
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

    agent_cartpole = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=False,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon_clip=0.2,
        value_clip=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        n_epochs=10,
        batch_size=64,
        hidden_dims=[64, 64]
    )

    history_cartpole = agent_cartpole.train(
        env=env_cartpole,
        n_episodes=300,
        max_steps=500,
        update_interval=2048,
        print_every=10,
        save_every=100,
        save_path='ppo_cartpole.pth'
    )

    plot_training_results(history_cartpole, 'ppo_cartpole_training.png')

    print("\nEvaluando política aprendida...")
    mean_reward, std_reward = evaluate_agent(agent_cartpole, env_cartpole, n_episodes=100)
    print(f"Recompensa promedio: {mean_reward:.2f} ± {std_reward:.2f}")

    env_cartpole.close()

    # ========== Ejemplo 2: LunarLander (Discreto) ==========
    print("\n" + "=" * 70)
    print("Ejemplo 2: LunarLander-v2 (Acciones Discretas)")
    print("=" * 70)

    env_lunar = gym.make('LunarLander-v2')
    state_dim_l = env_lunar.observation_space.shape[0]
    action_dim_l = env_lunar.action_space.n

    print(f"\nEntorno: LunarLander-v2")
    print(f"Estado: {state_dim_l} dimensiones")
    print(f"Acciones: {action_dim_l} (discretas)\n")

    agent_lunar = PPOAgent(
        state_dim=state_dim_l,
        action_dim=action_dim_l,
        continuous=False,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon_clip=0.2,
        value_clip=0.2,
        entropy_coef=0.01,
        n_epochs=4,
        batch_size=64,
        hidden_dims=[64, 64]
    )

    history_lunar = agent_lunar.train(
        env=env_lunar,
        n_episodes=1000,
        max_steps=1000,
        update_interval=2048,
        print_every=20,
        save_every=200,
        save_path='ppo_lunar.pth'
    )

    plot_training_results(history_lunar, 'ppo_lunar_training.png')

    print("\nEvaluando política aprendida...")
    mean_reward_l, std_reward_l = evaluate_agent(agent_lunar, env_lunar, n_episodes=50)
    print(f"Recompensa promedio: {mean_reward_l:.2f} ± {std_reward_l:.2f}")

    env_lunar.close()

    # ========== Ejemplo 3: Pendulum (Continuo) ==========
    print("\n" + "=" * 70)
    print("Ejemplo 3: Pendulum-v1 (Acciones Continuas)")
    print("=" * 70)

    env_pendulum = gym.make('Pendulum-v1')
    state_dim_p = env_pendulum.observation_space.shape[0]
    action_dim_p = env_pendulum.action_space.shape[0]

    print(f"\nEntorno: Pendulum-v1")
    print(f"Estado: {state_dim_p} dimensiones")
    print(f"Acciones: {action_dim_p} (continuas)\n")

    agent_pendulum = PPOAgent(
        state_dim=state_dim_p,
        action_dim=action_dim_p,
        continuous=True,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon_clip=0.2,
        value_clip=None,  # Sin value clipping para continuo
        entropy_coef=0.0,  # Sin entropy bonus para continuo
        n_epochs=10,
        batch_size=64,
        hidden_dims=[64, 64]
    )

    history_pendulum = agent_pendulum.train(
        env=env_pendulum,
        n_episodes=300,
        max_steps=200,
        update_interval=2048,
        print_every=10,
        save_every=100,
        save_path='ppo_pendulum.pth'
    )

    plot_training_results(history_pendulum, 'ppo_pendulum_training.png')

    print("\nEvaluando política aprendida...")
    mean_reward_p, std_reward_p = evaluate_agent(agent_pendulum, env_pendulum, n_episodes=50)
    print(f"Recompensa promedio: {mean_reward_p:.2f} ± {std_reward_p:.2f}")

    env_pendulum.close()

    print("\n" + "=" * 70)
    print("VENTAJAS DE PPO")
    print("=" * 70)
    print("1. Simple y confiable - fácil de implementar y tunear")
    print("2. Sample efficient - reutiliza datos con múltiples epochs")
    print("3. Stable - clipping previene updates demasiado grandes")
    print("4. Versatile - funciona bien en diversos ambientes")
    print("5. SOTA - usado en muchas aplicaciones reales (OpenAI, DeepMind)")
    print("=" * 70)


if __name__ == "__main__":
    main()
