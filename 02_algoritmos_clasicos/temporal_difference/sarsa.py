"""
SARSA (State-Action-Reward-State-Action) Algorithm

SARSA es un algoritmo on-policy TD control. A diferencia de Q-Learning,
actualiza usando la acción que realmente tomó (según su política actual),
no la mejor acción posible.

Algoritmo:
    Inicializar Q(s,a) arbitrariamente
    Para cada episodio:
        Inicializar s
        Elegir a desde s usando política derivada de Q (ej: ε-greedy)
        Para cada step del episodio:
            Tomar acción a, observar r, s'
            Elegir a' desde s' usando política derivada de Q
            Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
            s ← s'; a ← a'
        hasta que s sea terminal
"""

import numpy as np
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt


class SARSAAgent:
    """
    Agente SARSA (on-policy TD control)
    
    Parámetros:
        alpha: Tasa de aprendizaje
        gamma: Factor de descuento
        epsilon: Parámetro de exploración
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
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action, done):
        """
        Actualización SARSA
        
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        
        Nota: Usa la acción a' que realmente se tomará (on-policy)
        """
        current_q = self.Q[state][action]
        
        if done:
            target_q = reward
        else:
            # SARSA: usa Q(s', a') donde a' es la acción seleccionada por la política
            target_q = reward + self.gamma * self.Q[next_state][next_action]
        
        td_error = target_q - current_q
        self.Q[state][action] = current_q + self.alpha * td_error
    
    def decay_epsilon(self):
        """Decae epsilon después de cada episodio"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class ExpectedSARSAAgent:
    """
    Expected SARSA Agent
    
    Usa la esperanza sobre todas las acciones en lugar de una acción específica:
    Q(s,a) ← Q(s,a) + α[r + γ Σ_a' π(a'|s') Q(s',a') - Q(s,a)]
    
    Es más estable que SARSA y puede ser on-policy u off-policy.
    """
    
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.Q = defaultdict(lambda: np.zeros(n_actions))
    
    def get_action(self, state, training=True):
        """Selecciona acción usando ε-greedy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Actualización Expected SARSA
        
        Usa la esperanza sobre la política en lugar de una acción específica
        """
        current_q = self.Q[state][action]
        
        if done:
            target_q = reward
        else:
            # Calcular valor esperado bajo política ε-greedy
            best_action = np.argmax(self.Q[next_state])
            expected_q = 0
            
            for a in range(self.n_actions):
                if a == best_action:
                    # Probabilidad de acción greedy
                    prob = 1 - self.epsilon + self.epsilon / self.n_actions
                else:
                    # Probabilidad de acción exploratoria
                    prob = self.epsilon / self.n_actions
                
                expected_q += prob * self.Q[next_state][a]
            
            target_q = reward + self.gamma * expected_q
        
        td_error = target_q - current_q
        self.Q[state][action] = current_q + self.alpha * td_error
    
    def decay_epsilon(self):
        """Decae epsilon después de cada episodio"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_sarsa(env, agent, n_episodes=1000, max_steps=200):
    """
    Entrena agente SARSA
    
    Args:
        env: Ambiente gym
        agent: SARSAAgent o ExpectedSARSAAgent
        n_episodes: Número de episodios
        max_steps: Máximo de steps por episodio
    
    Returns:
        rewards_history: Lista de recompensas por episodio
    """
    rewards_history = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        
        # Convertir estado a hashable si es necesario
        if isinstance(state, np.ndarray):
            state = tuple(state) if state.size < 10 else hash(state.tobytes())
        
        # Seleccionar primera acción
        action = agent.get_action(state)
        
        episode_reward = 0
        
        for step in range(max_steps):
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if isinstance(next_state, np.ndarray):
                next_state = tuple(next_state) if next_state.size < 10 else hash(next_state.tobytes())
            
            episode_reward += reward
            
            if isinstance(agent, SARSAAgent):
                # SARSA: Seleccionar próxima acción antes de actualizar
                next_action = agent.get_action(next_state)
                agent.update(state, action, reward, next_state, next_action, done)
                action = next_action
            else:
                # Expected SARSA: No necesita próxima acción
                agent.update(state, action, reward, next_state, done)
                action = agent.get_action(next_state)
            
            state = next_state
            
            if done:
                break
        
        agent.decay_epsilon()
        rewards_history.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode + 1}/{n_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history


def compare_algorithms():
    """
    Compara SARSA, Expected SARSA y Q-Learning
    """
    env = gym.make('FrozenLake-v1', is_slippery=True)  # Ambiente estocástico
    
    n_episodes = 2000
    n_runs = 5  # Múltiples runs para promediar
    
    algorithms = {
        'SARSA': SARSAAgent,
        'Expected SARSA': ExpectedSARSAAgent,
    }
    
    results = {}
    
    for name, AgentClass in algorithms.items():
        print(f"\nEntrenando {name}...")
        all_rewards = []
        
        for run in range(n_runs):
            agent = AgentClass(
                n_actions=env.action_space.n,
                alpha=0.1,
                gamma=0.99,
                epsilon=1.0,
                epsilon_decay=0.995
            )
            
            rewards = train_sarsa(env, agent, n_episodes=n_episodes)
            all_rewards.append(rewards)
        
        # Promediar sobre runs
        results[name] = np.mean(all_rewards, axis=0)
    
    # Visualizar comparación
    plt.figure(figsize=(12, 6))
    
    window = 50
    for name, rewards in results.items():
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=name, linewidth=2)
    
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa promedio')
    plt.title('Comparación de Algoritmos TD Control')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sarsa_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nComparación guardada en 'sarsa_comparison.png'")
    env.close()


def main():
    """Ejemplo de uso: Entrenar SARSA en FrozenLake"""
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    # Crear agente Expected SARSA (generalmente mejor que SARSA)
    agent = ExpectedSARSAAgent(
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995
    )
    
    print("Entrenando Expected SARSA en FrozenLake...")
    print(f"Estados: {env.observation_space.n}")
    print(f"Acciones: {env.action_space.n}")
    print()
    
    rewards_history = train_sarsa(env, agent, n_episodes=2000)
    
    print(f"\nRecompensa promedio últimos 100 episodios: {np.mean(rewards_history[-100:]):.2f}")
    
    env.close()


if __name__ == "__main__":
    # Entrenar un agente
    main()
    
    # O comparar algoritmos
    # compare_algorithms()
