"""
REINFORCE: Monte Carlo Policy Gradient
========================================

Implementación del algoritmo REINFORCE (Williams, 1992), el algoritmo de Policy Gradient
más básico que usa retornos de Monte Carlo para actualizar directamente la política.

REINFORCE aprende una política paramétrica π_θ(a|s) mediante gradiente ascendente
en la función objetivo J(θ) = E[R|π_θ].

El gradiente se estima usando:
∇J(θ) = E[∇log π_θ(a|s) * G_t]

donde G_t es el retorno desde el tiempo t.

Características:
- On-policy: Aprende de su propia experiencia
- Monte Carlo: Usa episodios completos
- Alta varianza: Se mitiga con baseline
- Convergencia garantizada a mínimo local

Variantes implementadas:
1. REINFORCE básico
2. REINFORCE con baseline (value function)
3. Soporte para espacios discretos y continuos

Paper: "Simple Statistical Gradient-Following Algorithms for Connectionist
        Reinforcement Learning" (Williams, 1992)

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


class PolicyNetwork(nn.Module):
    """
    Red neuronal para la política π_θ(a|s).

    Para acciones discretas: Salida = logits sobre acciones (usamos Categorical)
    Para acciones continuas: Salida = media y log_std (usamos Normal)

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
                 hidden_dims: List[int] = [128, 128],
                 continuous: bool = False):
        super(PolicyNetwork, self).__init__()

        self.continuous = continuous
        self.action_dim = action_dim

        # Construir capas ocultas
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        if continuous:
            # Para continuo: salida = (mean, log_std)
            self.mean_layer = nn.Linear(prev_dim, action_dim)
            self.log_std_layer = nn.Linear(prev_dim, action_dim)
        else:
            # Para discreto: salida = logits
            self.action_head = nn.Linear(prev_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            state: Tensor de estados [batch_size, state_dim]

        Returns:
            Si discreto: (logits, None)
            Si continuo: (mean, log_std)
        """
        x = self.shared_layers(state)

        if self.continuous:
            mean = self.mean_layer(x)
            log_std = self.log_std_layer(x)
            # Limitar log_std para estabilidad
            log_std = torch.clamp(log_std, -20, 2)
            return mean, log_std
        else:
            logits = self.action_head(x)
            return logits, None

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        Selecciona acción dado un estado.

        Args:
            state: Estado [state_dim]
            deterministic: Si True, retorna acción determinista (mean para continuo, argmax para discreto)

        Returns:
            action: Acción seleccionada
            log_prob: Log probabilidad de la acción
            entropy: Entropía de la distribución (para regularización)
        """
        if self.continuous:
            mean, log_std = self.forward(state)
            std = log_std.exp()

            if deterministic:
                action = mean
                # Para acciones deterministas, log_prob no aplica
                return action, None, None

            # Distribución normal
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions
            entropy = dist.entropy().sum(dim=-1)

            return action, log_prob, entropy
        else:
            logits, _ = self.forward(state)

            if deterministic:
                action = logits.argmax(dim=-1)
                return action, None, None

            # Distribución categórica
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            return action, log_prob, entropy


class ValueNetwork(nn.Module):
    """
    Red neuronal para la función de valor V(s) (baseline).

    El baseline reduce la varianza del estimador de gradiente sin introducir sesgo.
    V(s) aproxima el valor esperado desde el estado s.

    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    hidden_dims : List[int]
        Dimensiones de capas ocultas
    """

    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 128]):
        super(ValueNetwork, self).__init__()

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
            state: Tensor de estados [batch_size, state_dim]

        Returns:
            values: Valores estimados [batch_size]
        """
        return self.network(state).squeeze(-1)


class REINFORCEAgent:
    """
    Agente REINFORCE con baseline opcional.

    Implementa el algoritmo REINFORCE con soporte para:
    - Acciones discretas y continuas
    - Baseline (value function) para reducir varianza
    - Regularización de entropía para mejorar exploración
    - Normalización de ventajas

    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool
        True para acciones continuas
    learning_rate : float
        Tasa de aprendizaje de la política
    gamma : float
        Factor de descuento
    use_baseline : bool
        Si True, usa value network como baseline
    baseline_lr : float
        Tasa de aprendizaje del baseline
    entropy_coef : float
        Coeficiente de regularización de entropía
    normalize_advantages : bool
        Si True, normaliza las ventajas (reduce varianza)
    hidden_dims : List[int]
        Dimensiones de capas ocultas
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        continuous: bool = False,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        use_baseline: bool = True,
        baseline_lr: float = 1e-3,
        entropy_coef: float = 0.01,
        normalize_advantages: bool = True,
        hidden_dims: List[int] = [128, 128]
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.entropy_coef = entropy_coef
        self.normalize_advantages = normalize_advantages

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Redes neuronales
        self.policy = PolicyNetwork(
            state_dim, action_dim, hidden_dims, continuous
        ).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Baseline (value network)
        if use_baseline:
            self.value_network = ValueNetwork(state_dim, hidden_dims).to(self.device)
            self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=baseline_lr)
        else:
            self.value_network = None

        # Historial
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Selecciona acción para un estado.

        Args:
            state: Estado actual (numpy array)
            deterministic: Si True, acción determinista

        Returns:
            action: Acción seleccionada (numpy array o int)
            log_prob: Log probabilidad (solo si no deterministic)
            entropy: Entropía (solo si no deterministic)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, entropy = self.policy.get_action(state_tensor, deterministic)

        if self.continuous:
            action = action.cpu().numpy().flatten()
        else:
            action = action.item()

        if deterministic:
            return action

        return action, log_prob.item(), entropy.item() if entropy is not None else 0.0

    def compute_returns(self, rewards: List[float]) -> List[float]:
        """
        Calcula retornos descontados G_t para cada paso.

        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...

        Args:
            rewards: Lista de recompensas del episodio

        Returns:
            returns: Lista de retornos para cada paso
        """
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    def train_episode(self, states: List[np.ndarray], actions: List,
                     log_probs: List[float], entropies: List[float],
                     rewards: List[float]) -> Dict[str, float]:
        """
        Entrena la política usando un episodio completo.

        REINFORCE update:
        ∇J(θ) ≈ Σ_t ∇log π_θ(a_t|s_t) * (G_t - b(s_t))

        Args:
            states: Lista de estados
            actions: Lista de acciones
            log_probs: Lista de log probabilidades
            entropies: Lista de entropías
            rewards: Lista de recompensas

        Returns:
            metrics: Diccionario con losses y métricas
        """
        # Calcular retornos
        returns = self.compute_returns(rewards)

        # Convertir a tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        entropies_tensor = torch.FloatTensor(entropies).to(self.device)

        # Calcular baseline y ventajas
        if self.use_baseline:
            values = self.value_network(states_tensor)
            advantages = returns_tensor - values.detach()

            # Entrenar value network
            value_loss = F.mse_loss(values, returns_tensor)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
            self.value_optimizer.step()
        else:
            advantages = returns_tensor
            value_loss = torch.tensor(0.0)

        # Normalizar ventajas (reduce varianza)
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss: -E[log π(a|s) * A(s,a)]
        policy_loss = -(log_probs_tensor * advantages).mean()

        # Entropy bonus (fomenta exploración)
        entropy_loss = -entropies_tensor.mean()

        # Loss total
        total_loss = policy_loss + self.entropy_coef * entropy_loss

        # Optimizar política
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item() if self.use_baseline else 0.0,
            'entropy': entropies_tensor.mean().item(),
            'total_loss': total_loss.item()
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
        Entrena el agente REINFORCE.

        Args:
            env: Entorno gymnasium
            n_episodes: Número de episodios
            max_steps: Pasos máximos por episodio
            print_every: Frecuencia de logging
            render: Si True, renderiza el entorno

        Returns:
            history: Diccionario con historial de entrenamiento
        """
        print(f"Entrenando REINFORCE {'con' if self.use_baseline else 'sin'} baseline...")
        print(f"Episodios: {n_episodes}, Gamma: {self.gamma}, "
              f"Entropy coef: {self.entropy_coef}\n")

        for episode in range(n_episodes):
            # Resetear episodio
            state, _ = env.reset()

            # Almacenar trayectoria
            states = []
            actions_list = []
            log_probs = []
            entropies = []
            rewards = []

            # Generar episodio
            for step in range(max_steps):
                # Seleccionar acción
                action, log_prob, entropy = self.get_action(state, deterministic=False)

                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Almacenar transición
                states.append(state)
                actions_list.append(action)
                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards.append(reward)

                state = next_state

                if done:
                    break

            # Entrenar con el episodio
            metrics = self.train_episode(states, actions_list, log_probs,
                                        entropies, rewards)

            # Registrar estadísticas
            episode_reward = sum(rewards)
            self.history['episode_rewards'].append(episode_reward)
            self.history['episode_lengths'].append(len(rewards))
            self.history['policy_losses'].append(metrics['policy_loss'])
            self.history['value_losses'].append(metrics['value_loss'])
            self.history['entropies'].append(metrics['entropy'])

            # Logging
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.history['episode_rewards'][-100:])
                avg_length = np.mean(self.history['episode_lengths'][-100:])
                avg_entropy = np.mean(self.history['entropies'][-100:])

                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg Reward (100): {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Entropy: {avg_entropy:.3f} | "
                      f"Policy Loss: {metrics['policy_loss']:.4f}")

        print("\nEntrenamiento completado!")
        return self.history

    def save(self, path: str):
        """Guarda el modelo."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_network.state_dict() if self.value_network else None,
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict() if self.use_baseline else None,
            'history': self.history
        }, path)
        print(f"Modelo guardado en {path}")

    def load(self, path: str):
        """Carga el modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if self.use_baseline and checkpoint['value_state_dict']:
            self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.history = checkpoint['history']
        print(f"Modelo cargado desde {path}")


def plot_training_results(history: Dict[str, List], save_path: str = 'reinforce_training.png'):
    """
    Visualiza resultados del entrenamiento.

    Args:
        history: Historial de entrenamiento
        save_path: Ruta donde guardar la figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

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

    # Policy Loss
    ax = axes[1, 0]
    losses = history['policy_losses']
    ax.plot(losses, alpha=0.3, label='Policy Loss')
    if len(losses) >= 10:
        window = min(100, len(losses) // 10)
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(losses)), moving_avg,
               label=f'MA({window})', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Policy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropía
    ax = axes[1, 1]
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráficos guardados en '{save_path}'")


def evaluate_agent(agent: REINFORCEAgent, env: gym.Env,
                   n_episodes: int = 10, render: bool = False) -> Tuple[float, float]:
    """
    Evalúa el agente entrenado.

    Args:
        agent: Agente REINFORCE
        env: Entorno
        n_episodes: Número de episodios de evaluación
        render: Si True, renderiza

    Returns:
        mean_reward: Recompensa promedio
        std_reward: Desviación estándar de recompensa
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
    """Ejemplo de uso: Entrenar REINFORCE en CartPole-v1"""

    print("=" * 70)
    print("REINFORCE - Ejemplo con CartPole-v1")
    print("=" * 70)

    # Crear entorno
    env = gym.make('CartPole-v1')

    # Dimensiones
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"\nEntorno: CartPole-v1")
    print(f"Estado: {state_dim} dimensiones")
    print(f"Acciones: {action_dim} (discretas)")
    print()

    # ========== REINFORCE con Baseline ==========
    print("\n" + "=" * 70)
    print("Entrenando REINFORCE con Baseline")
    print("=" * 70)

    agent_baseline = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=False,
        learning_rate=3e-4,
        gamma=0.99,
        use_baseline=True,
        baseline_lr=1e-3,
        entropy_coef=0.01,
        normalize_advantages=True,
        hidden_dims=[128, 128]
    )

    history_baseline = agent_baseline.train(
        env=env,
        n_episodes=500,
        max_steps=500,
        print_every=50
    )

    # Visualizar
    plot_training_results(history_baseline, 'reinforce_baseline_training.png')

    # Evaluar
    print("\nEvaluando política aprendida...")
    mean_reward, std_reward = evaluate_agent(agent_baseline, env, n_episodes=100)
    print(f"Recompensa promedio: {mean_reward:.2f} ± {std_reward:.2f}")

    # ========== REINFORCE sin Baseline ==========
    print("\n" + "=" * 70)
    print("Entrenando REINFORCE sin Baseline (comparación)")
    print("=" * 70)

    agent_no_baseline = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=False,
        learning_rate=3e-4,
        gamma=0.99,
        use_baseline=False,
        entropy_coef=0.01,
        normalize_advantages=True,
        hidden_dims=[128, 128]
    )

    history_no_baseline = agent_no_baseline.train(
        env=env,
        n_episodes=500,
        max_steps=500,
        print_every=50
    )

    # Visualizar
    plot_training_results(history_no_baseline, 'reinforce_no_baseline_training.png')

    # Evaluar
    mean_reward_nb, std_reward_nb = evaluate_agent(agent_no_baseline, env, n_episodes=100)
    print(f"Recompensa promedio: {mean_reward_nb:.2f} ± {std_reward_nb:.2f}")

    # ========== Comparación ==========
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE RESULTADOS")
    print("=" * 70)
    print(f"REINFORCE con Baseline:    {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"REINFORCE sin Baseline:    {mean_reward_nb:.2f} ± {std_reward_nb:.2f}")
    print("\nConclusión: El baseline reduce la varianza y mejora el aprendizaje.")
    print("=" * 70)

    # Guardar modelo
    agent_baseline.save('reinforce_cartpole.pth')

    env.close()


if __name__ == "__main__":
    main()
