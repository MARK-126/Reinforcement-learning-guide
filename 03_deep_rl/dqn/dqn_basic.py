"""
Deep Q-Network (DQN) Implementation

DQN combina Q-Learning con redes neuronales profundas.
Incluye dos técnicas clave:
1. Experience Replay: Almacena transiciones y entrena con mini-batches aleatorios
2. Target Network: Red separada para calcular targets, actualizada periódicamente

Paper: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt


# Transición para replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    Experience Replay Buffer
    
    Almacena transiciones (s, a, r, s', done) y permite muestreo aleatorio.
    Esto rompe la correlación temporal de las experiencias.
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Añade una transición al buffer"""
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Muestrea un batch aleatorio de transiciones"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """
    Red Neuronal para Q-Network
    
    Arquitectura simple: MLP con 2 capas ocultas
    Input: Estado
    Output: Q-values para cada acción
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        """Forward pass: estado -> Q-values"""
        return self.network(x)


class DQNAgent:
    """
    Agente DQN
    
    Parámetros:
        state_dim: Dimensión del espacio de estados
        action_dim: Número de acciones
        learning_rate: Tasa de aprendizaje
        gamma: Factor de descuento
        epsilon_start: Epsilon inicial
        epsilon_end: Epsilon final
        epsilon_decay: Pasos para decaer epsilon
        buffer_size: Tamaño del replay buffer
        batch_size: Tamaño del mini-batch
        target_update: Frecuencia de actualización de target network
    """
    
    def __init__(self, state_dim, action_dim, 
                 learning_rate=1e-3,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=1000,
                 buffer_size=10000,
                 batch_size=64,
                 target_update=10):
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Q-Network y Target Network
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network siempre en eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Contador de steps
        self.steps = 0
    
    def get_action(self, state, training=True):
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
    
    def update_epsilon(self):
        """Actualiza epsilon con decaimiento exponencial"""
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-self.steps / self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Almacena transición en replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Realiza un paso de entrenamiento
        
        1. Muestrea mini-batch del replay buffer
        2. Calcula Q-values actuales
        3. Calcula targets usando target network
        4. Calcula loss y actualiza Q-network
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
        
        # Q-values actuales: Q(s, a)
        current_q_values = self.q_network(state_batch).gather(1, action_batch)
        
        # Targets: r + γ max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
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
    
    def update_target_network(self):
        """Copia pesos de Q-network a target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())


def train_dqn(env, agent, n_episodes=500, max_steps=500, render=False):
    """
    Entrena agente DQN
    
    Args:
        env: Ambiente gymnasium
        agent: DQNAgent
        n_episodes: Número de episodios
        max_steps: Máximo de steps por episodio
        render: Si True, renderiza el ambiente
    
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
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Actualizar target network periódicamente
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
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
    
    return rewards_history, losses_history


def plot_training(rewards, losses):
    """Visualiza resultados del entrenamiento"""
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
    ax1.set_title('DQN Training Rewards')
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
    ax2.set_title('DQN Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráficos guardados en 'dqn_training.png'")


def main():
    """Ejemplo de uso: Entrenar DQN en CartPole"""
    # Crear ambiente
    env = gym.make('CartPole-v1')
    
    # Obtener dimensiones
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Estado: {state_dim} dimensiones")
    print(f"Acciones: {action_dim}")
    print()
    
    # Crear agente
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
        buffer_size=10000,
        batch_size=64,
        target_update=10
    )
    
    # Entrenar
    print("Entrenando DQN en CartPole...")
    rewards, losses = train_dqn(env, agent, n_episodes=300, max_steps=500)
    
    # Visualizar
    plot_training(rewards, losses)
    
    print(f"\nReward promedio últimos 10 episodios: {np.mean(rewards[-10:]):.2f}")
    
    env.close()


if __name__ == "__main__":
    main()
