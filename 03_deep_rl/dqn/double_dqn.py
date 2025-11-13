"""
Double Deep Q-Network (Double DQN) Implementation

Double DQN mejora el DQN estándar al reducir el sesgo de sobreestimación.
Usa la red online para seleccionar acciones y la red target para evaluarlas.

Diferencia clave:
- DQN estándar: target = r + γ max_a' Q_target(s', a')
- Double DQN: target = r + γ Q_target(s', argmax_a' Q_online(s', a'))

Paper: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
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


class DQN(nn.Module):
    """
    Red Neuronal para Q-Network

    Arquitectura: MLP con 2 capas ocultas
    Input: Estado
    Output: Q-values para cada acción

    Args:
        state_dim: Dimensión del espacio de estados
        action_dim: Número de acciones discretas
        hidden_dim: Dimensión de capas ocultas
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: estado -> Q-values"""
        return self.network(x)


class DoubleDQNAgent:
    """
    Agente Double DQN

    Implementa Double Q-learning para reducir sobreestimación de Q-values.
    Usa la red online para seleccionar acciones y la target para evaluarlas.

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
                 hidden_dim: int = 128):

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.tau = tau  # Si no es None, usa soft update

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Q-Network (online) y Target Network
        self.q_network = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network siempre en eval mode

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
        Realiza un paso de entrenamiento con Double DQN

        Double DQN update:
        1. Muestrea mini-batch del replay buffer
        2. Calcula Q-values actuales: Q_online(s, a)
        3. Selecciona mejores acciones con online network: a* = argmax_a Q_online(s', a)
        4. Evalúa con target network: target = r + γ Q_target(s', a*)
        5. Calcula loss y actualiza Q-network

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

        # Q-values actuales: Q_online(s, a)
        current_q_values = self.q_network(state_batch).gather(1, action_batch)

        # Double DQN: seleccionar con online, evaluar con target
        with torch.no_grad():
            # Seleccionar mejores acciones con online network
            next_actions = self.q_network(next_state_batch).argmax(1, keepdim=True)
            # Evaluar esas acciones con target network
            next_q_values = self.target_network(next_state_batch).gather(1, next_actions).squeeze()
            # Calcular targets
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Loss: MSE entre Q actual y target
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping para estabilidad
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
            'episodes': self.episodes
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
        print(f"Modelo cargado desde {path}")


def train_double_dqn(env: gym.Env,
                     agent: DoubleDQNAgent,
                     n_episodes: int = 500,
                     max_steps: int = 500,
                     save_every: int = 100,
                     save_path: str = "double_dqn_model.pth") -> Tuple[List[float], List[float]]:
    """
    Entrena agente Double DQN

    Args:
        env: Ambiente gymnasium
        agent: DoubleDQNAgent
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
                   agent: DoubleDQNAgent,
                   n_episodes: int = 10,
                   render: bool = False) -> float:
    """
    Evalúa el agente sin exploración

    Args:
        env: Ambiente gymnasium
        agent: DoubleDQNAgent
        n_episodes: Número de episodios de evaluación
        render: Si True, renderiza el ambiente

    Returns:
        Recompensa promedio
    """
    eval_rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            if render:
                env.render()

            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state

        eval_rewards.append(episode_reward)

    return np.mean(eval_rewards)


def plot_training(rewards: List[float],
                 losses: List[float],
                 save_path: str = "double_dqn_training.png") -> None:
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
    ax1.set_title('Double DQN Training Rewards')
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
    ax2.set_title('Double DQN Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráficos guardados en '{save_path}'")


def compare_with_standard_dqn(rewards_ddqn: List[float],
                              rewards_dqn: Optional[List[float]] = None,
                              save_path: str = "ddqn_vs_dqn.png") -> None:
    """
    Compara Double DQN con DQN estándar

    Args:
        rewards_ddqn: Recompensas de Double DQN
        rewards_dqn: Recompensas de DQN estándar (opcional)
        save_path: Ruta para guardar la figura
    """
    plt.figure(figsize=(10, 6))

    # Double DQN
    window = 10
    if len(rewards_ddqn) >= window:
        moving_avg_ddqn = np.convolve(rewards_ddqn, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards_ddqn)), moving_avg_ddqn,
                label='Double DQN', linewidth=2, color='blue')

    # DQN estándar (si se proporciona)
    if rewards_dqn is not None and len(rewards_dqn) >= window:
        moving_avg_dqn = np.convolve(rewards_dqn, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards_dqn)), moving_avg_dqn,
                label='Standard DQN', linewidth=2, color='red', alpha=0.7)

    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Double DQN vs Standard DQN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparación guardada en '{save_path}'")


def main():
    """
    Ejemplo de uso: Entrenar Double DQN en CartPole o LunarLander

    Double DQN mejora DQN estándar al reducir sobreestimación de Q-values.
    Esto resulta en aprendizaje más estable y mejor rendimiento.
    """
    # Seleccionar ambiente
    # env_name = 'CartPole-v1'  # Más simple
    env_name = 'LunarLander-v2'  # Más complejo

    print(f"=== Entrenando Double DQN en {env_name} ===\n")

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
            'n_episodes': 600,
            'max_steps': 1000
        }

    # Crear agente
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **{k: v for k, v in config.items() if k not in ['n_episodes', 'max_steps']}
    )

    print("Arquitectura:")
    print(agent.q_network)
    print()

    # Entrenar
    print(f"Entrenando Double DQN durante {config['n_episodes']} episodios...")
    print(f"Update strategy: {'Soft update (tau={})'.format(config['tau']) if config['tau'] else 'Hard update'}")
    print()

    rewards, losses = train_double_dqn(
        env, agent,
        n_episodes=config['n_episodes'],
        max_steps=config['max_steps'],
        save_every=100,
        save_path=f"double_dqn_{env_name.lower()}.pth"
    )

    # Visualizar
    plot_training(rewards, losses, save_path=f"double_dqn_{env_name.lower()}_training.png")

    # Evaluación final
    print("\n" + "="*50)
    print("Evaluación final (sin exploración):")
    avg_reward = evaluate_agent(env, agent, n_episodes=20)
    print(f"Recompensa promedio: {avg_reward:.2f}")

    print(f"\nReward promedio últimos 10 episodios entrenamiento: {np.mean(rewards[-10:]):.2f}")
    print(f"Reward promedio últimos 50 episodios entrenamiento: {np.mean(rewards[-50:]):.2f}")

    # Guardar modelo final
    agent.save(f"double_dqn_{env_name.lower()}_final.pth")

    env.close()

    print("\n" + "="*50)
    print("Ventajas de Double DQN:")
    print("1. Reduce sobreestimación de Q-values")
    print("2. Aprendizaje más estable")
    print("3. Mejor rendimiento en ambientes complejos")
    print("4. Mismo costo computacional que DQN estándar")


if __name__ == "__main__":
    main()
