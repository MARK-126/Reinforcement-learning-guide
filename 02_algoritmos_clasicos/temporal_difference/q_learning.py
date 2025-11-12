"""
Q-Learning Algorithm Implementation

Q-Learning es un algoritmo off-policy TD control que aprende la función Q* óptima.
No requiere un modelo del ambiente y es uno de los algoritmos más importantes en RL.

Algoritmo:
    Inicializar Q(s,a) arbitrariamente
    Para cada episodio:
        Inicializar s
        Para cada step del episodio:
            Elegir a desde s usando política derivada de Q (ej: ε-greedy)
            Tomar acción a, observar r, s'
            Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
            s ← s'
        hasta que s sea terminal
"""

import numpy as np
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt


class QLearningAgent:
    """
    Agente Q-Learning
    
    Parámetros:
        alpha: Tasa de aprendizaje (learning rate)
        gamma: Factor de descuento
        epsilon: Parámetro de exploración para ε-greedy
        epsilon_decay: Factor de decaimiento para epsilon
        epsilon_min: Valor mínimo de epsilon
    """
    
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Inicializar Q-table
        self.Q = defaultdict(lambda: np.zeros(n_actions))
    
    def get_action(self, state, training=True):
        """
        Selecciona acción usando ε-greedy
        
        Args:
            state: Estado actual
            training: Si False, siempre selecciona acción greedy
        
        Returns:
            Acción seleccionada
        """
        if training and np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return np.random.randint(self.n_actions)
        else:
            # Explotación: mejor acción según Q
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Actualización de Q-Learning
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        # Valor actual
        current_q = self.Q[state][action]
        
        # Valor objetivo (TD target)
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.Q[next_state])
        
        # Error TD
        td_error = target_q - current_q
        
        # Actualización
        self.Q[state][action] = current_q + self.alpha * td_error
    
    def decay_epsilon(self):
        """Decaer epsilon después de cada episodio"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def discretize_state(state, bins):
    """
    Discretiza estados continuos en bins
    
    Args:
        state: Estado continuo (array)
        bins: Lista de bins para cada dimensión
    
    Returns:
        Estado discretizado (tupla)
    """
    state_adj = []
    for i, s in enumerate(state):
        state_adj.append(np.digitize(s, bins[i]) - 1)
    return tuple(state_adj)


def train_q_learning(env, agent, n_episodes=1000, max_steps=200):
    """
    Entrena agente Q-Learning
    
    Args:
        env: Ambiente gym
        agent: Agente QLearningAgent
        n_episodes: Número de episodios
        max_steps: Máximo de steps por episodio
    
    Returns:
        rewards_history: Lista de recompensas por episodio
        steps_history: Lista de steps por episodio
    """
    rewards_history = []
    steps_history = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        
        # Discretizar si es necesario
        if isinstance(state, np.ndarray):
            # Para ambientes como CartPole, crear bins
            state = tuple(state) if state.size < 10 else hash(state.tobytes())
        
        episode_reward = 0
        
        for step in range(max_steps):
            # Seleccionar acción
            action = agent.get_action(state)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Procesar próximo estado
            if isinstance(next_state, np.ndarray):
                next_state = tuple(next_state) if next_state.size < 10 else hash(next_state.tobytes())
            
            # Actualizar Q-values
            agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Decaer epsilon
        agent.decay_epsilon()
        
        rewards_history.append(episode_reward)
        steps_history.append(step + 1)
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode + 1}/{n_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history, steps_history


def evaluate_agent(env, agent, n_episodes=10):
    """
    Evalúa agente entrenado
    
    Args:
        env: Ambiente gym
        agent: Agente QLearningAgent entrenado
        n_episodes: Número de episodios de evaluación
    
    Returns:
        Recompensa promedio
    """
    total_rewards = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        if isinstance(state, np.ndarray):
            state = tuple(state) if state.size < 10 else hash(state.tobytes())
        
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if isinstance(next_state, np.ndarray):
                next_state = tuple(next_state) if next_state.size < 10 else hash(next_state.tobytes())
            
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


def plot_results(rewards_history, steps_history):
    """Visualiza resultados del entrenamiento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Recompensas
    ax1.plot(rewards_history, alpha=0.6, label='Reward por episodio')
    
    # Promedio móvil
    window = 50
    if len(rewards_history) >= window:
        moving_avg = np.convolve(rewards_history, 
                                 np.ones(window)/window, 
                                 mode='valid')
        ax1.plot(range(window-1, len(rewards_history)), 
                moving_avg, 
                label=f'Promedio móvil ({window} eps)',
                linewidth=2)
    
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Recompensas durante entrenamiento')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Steps
    ax2.plot(steps_history, alpha=0.6)
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Steps')
    ax2.set_title('Steps por episodio')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('q_learning_results.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """
    Ejemplo de uso: Entrenar Q-Learning en FrozenLake
    """
    # Crear ambiente
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    # Crear agente
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print("Entrenando Q-Learning en FrozenLake...")
    print(f"Estados: {env.observation_space.n}")
    print(f"Acciones: {env.action_space.n}")
    print()
    
    # Entrenar
    rewards_history, steps_history = train_q_learning(
        env, agent, 
        n_episodes=2000,
        max_steps=100
    )
    
    # Evaluar
    print("\nEvaluando agente...")
    avg_reward = evaluate_agent(env, agent, n_episodes=100)
    print(f"Recompensa promedio en evaluación: {avg_reward:.2f}")
    
    # Visualizar
    plot_results(rewards_history, steps_history)
    print("\nGráficos guardados en 'q_learning_results.png'")
    
    env.close()


if __name__ == "__main__":
    main()
