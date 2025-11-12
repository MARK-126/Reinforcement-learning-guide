"""
CartPole con Q-Learning y Discretización

Este ejemplo muestra cómo resolver CartPole usando Q-Learning clásico.
Como CartPole tiene estados continuos, primero discretizamos el espacio de estados.
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os

# Añadir path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.plotting import plot_rewards


class CartPoleDiscretizer:
    """
    Discretiza el espacio de estados continuo de CartPole
    """
    
    def __init__(self, n_bins=10):
        """
        Args:
            n_bins: Número de bins por dimensión
        """
        # Límites aproximados del espacio de estados
        # [cart_pos, cart_vel, pole_angle, pole_vel]
        self.lower_bounds = np.array([-2.4, -3.0, -0.21, -3.0])
        self.upper_bounds = np.array([2.4, 3.0, 0.21, 3.0])
        
        # Crear bins para cada dimensión
        self.n_bins = n_bins
        self.bins = [
            np.linspace(low, high, n_bins - 1)
            for low, high in zip(self.lower_bounds, self.upper_bounds)
        ]
    
    def discretize(self, state):
        """
        Convierte estado continuo a discreto
        
        Args:
            state: Estado continuo (4D)
        
        Returns:
            Tupla de índices discretos
        """
        state_discrete = []
        for i, s in enumerate(state):
            # Clip state to bounds
            s = np.clip(s, self.lower_bounds[i], self.upper_bounds[i])
            # Find bin
            idx = np.digitize(s, self.bins[i])
            state_discrete.append(idx)
        
        return tuple(state_discrete)


class QLearningAgent:
    """
    Agente Q-Learning para CartPole discretizado
    """
    
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: defaultdict para manejar nuevos estados automáticamente
        self.Q = defaultdict(lambda: np.zeros(n_actions))
    
    def get_action(self, state, training=True):
        """Selecciona acción usando ε-greedy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        """Actualización Q-Learning"""
        current_q = self.Q[state][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.Q[next_state])
        
        self.Q[state][action] = current_q + self.alpha * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decae epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(env, agent, discretizer, n_episodes=500, max_steps=500):
    """
    Entrena agente Q-Learning en CartPole
    
    Returns:
        rewards_history: Lista de recompensas por episodio
        steps_history: Lista de steps por episodio
    """
    rewards_history = []
    steps_history = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        state = discretizer.discretize(state)
        
        episode_reward = 0
        
        for step in range(max_steps):
            # Seleccionar y ejecutar acción
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = discretizer.discretize(next_state)
            
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
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode + 1}/{n_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history, steps_history


def evaluate(env, agent, discretizer, n_episodes=100, render=False):
    """
    Evalúa agente entrenado
    
    Returns:
        Recompensa promedio
    """
    total_rewards = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        state = discretizer.discretize(state)
        
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = discretizer.discretize(next_state)
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    return avg_reward, std_reward


def visualize_policy(env, agent, discretizer, n_episodes=3):
    """
    Visualiza la política aprendida
    """
    for episode in range(n_episodes):
        state, _ = env.reset()
        state_discrete = discretizer.discretize(state)
        
        print(f"\n--- Episode {episode + 1} ---")
        done = False
        total_reward = 0
        step = 0
        
        while not done and step < 500:
            action = agent.get_action(state_discrete, training=False)
            action_name = "LEFT" if action == 0 else "RIGHT"
            
            print(f"Step {step}: State={state[:2]}, "  # Solo mostrar pos y vel
                  f"Action={action_name}, "
                  f"Q-values={agent.Q[state_discrete]}")
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            state_discrete = discretizer.discretize(state)
            total_reward += reward
            step += 1
        
        print(f"Total Reward: {total_reward}")


def main():
    """Función principal"""
    # Crear ambiente
    env = gym.make('CartPole-v1')
    
    # Crear discretizador
    discretizer = CartPoleDiscretizer(n_bins=10)
    
    # Crear agente
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print("="*50)
    print("CartPole con Q-Learning + Discretización")
    print("="*50)
    print(f"Espacio de estados: {env.observation_space.shape}")
    print(f"Espacio de acciones: {env.action_space.n}")
    print(f"Bins por dimensión: {discretizer.n_bins}")
    print(f"Total estados discretos: {discretizer.n_bins ** 4}")
    print()
    
    # Entrenar
    print("Entrenando...")
    rewards, steps = train(env, agent, discretizer, n_episodes=500)
    
    # Evaluar
    print("\n" + "="*50)
    print("Evaluación")
    print("="*50)
    avg_reward, std_reward = evaluate(env, agent, discretizer, n_episodes=100)
    print(f"Recompensa promedio: {avg_reward:.2f} ± {std_reward:.2f}")
    
    # Criterio de éxito: CartPole se considera "resuelto" si avg reward > 195
    if avg_reward >= 195:
        print("✓ ¡Problema resuelto!")
    else:
        print("✗ No alcanzó el criterio de éxito (195)")
    
    # Visualizar política
    print("\n" + "="*50)
    print("Ejemplos de Política Aprendida")
    print("="*50)
    visualize_policy(env, agent, discretizer, n_episodes=2)
    
    # Graficar resultados
    print("\nGenerando gráficos...")
    plot_rewards(rewards, window=20, 
                title="CartPole Q-Learning Training",
                save_path="cartpole_qlearning_results.png")
    
    print("\n✓ Entrenamiento completado!")
    
    env.close()


if __name__ == "__main__":
    main()
