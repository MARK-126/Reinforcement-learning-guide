"""
Lunar Lander con DQN
====================

Ejemplo completo de entrenamiento de un agente para LunarLander-v2
usando Deep Q-Network y sus variantes.

Autor: MARK-126
"""

import sys
sys.path.insert(0, '/home/user/Reinforcement-learning-guide')

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from deep_rl.dqn import DQNAgent, DoubleDQNAgent, DuelingDQNAgent


def train_agent(
    agent_type: str = 'dqn',
    n_episodes: int = 600,
    render: bool = False
) -> Tuple[object, List[float]]:
    """
    Entrena un agente en LunarLander-v2.

    Parámetros:
    -----------
    agent_type : str
        Tipo de agente: 'dqn', 'double_dqn', 'dueling_dqn'
    n_episodes : int
        Número de episodios de entrenamiento
    render : bool
        Si True, renderiza el ambiente

    Retorna:
    --------
    agent : object
        Agente entrenado
    rewards : List[float]
        Recompensas por episodio
    """
    # Crear ambiente
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]  # 8
    action_dim = env.action_space.n  # 4

    # Crear agente según el tipo
    if agent_type == 'dqn':
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=5e-4,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            buffer_size=100000,
            batch_size=64,
            target_update_freq=10
        )
    elif agent_type == 'double_dqn':
        agent = DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=5e-4,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            buffer_size=100000,
            batch_size=64,
            tau=0.001  # Soft update
        )
    elif agent_type == 'dueling_dqn':
        agent = DuelingDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            use_double_dqn=True,  # Combinar con Double DQN
            learning_rate=5e-4,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            buffer_size=100000,
            batch_size=64,
            tau=0.001
        )
    else:
        raise ValueError(f"Tipo de agente desconocido: {agent_type}")

    print(f"\n{'='*60}")
    print(f"Entrenando {agent_type.upper()} en LunarLander-v2")
    print(f"{'='*60}\n")

    rewards = []
    losses = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        episode_loss = []

        while not done:
            # Seleccionar acción
            action = agent.get_action(state)

            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Almacenar transición
            agent.store_transition(state, action, reward, next_state, done)

            # Entrenar
            if len(agent.replay_buffer) > agent.batch_size:
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)

            state = next_state
            total_reward += reward
            steps += 1

            if render and episode % 50 == 0:
                env.render()

        rewards.append(total_reward)
        if episode_loss:
            losses.append(np.mean(episode_loss))

        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"Episodio {episode+1}/{n_episodes} | "
                  f"Recompensa: {total_reward:.2f} | "
                  f"Promedio (10): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Steps: {steps}")

        # Criterio de éxito
        if len(rewards) >= 100 and np.mean(rewards[-100:]) >= 200:
            print(f"\n✓ ¡Problema resuelto en {episode+1} episodios!")
            print(f"  Promedio de últimos 100 episodios: {np.mean(rewards[-100:]):.2f}")
            break

    env.close()
    return agent, rewards, losses


def evaluate_agent(agent, n_episodes: int = 10, render: bool = False) -> float:
    """
    Evalúa un agente entrenado.

    Parámetros:
    -----------
    agent : object
        Agente a evaluar
    n_episodes : int
        Número de episodios de evaluación
    render : bool
        Si True, renderiza el ambiente

    Retorna:
    --------
    avg_reward : float
        Recompensa promedio
    """
    env = gym.make('LunarLander-v2', render_mode='human' if render else None)

    total_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Usar política greedy (sin exploración)
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episodio {episode+1}: {total_reward:.2f}")

    env.close()
    avg_reward = np.mean(total_rewards)
    print(f"\nRecompensa promedio: {avg_reward:.2f} ± {np.std(total_rewards):.2f}")

    return avg_reward


def compare_agents():
    """
    Compara el rendimiento de DQN, Double DQN y Dueling DQN.
    """
    n_episodes = 400
    results = {}

    for agent_type in ['dqn', 'double_dqn', 'dueling_dqn']:
        print(f"\n{'='*60}")
        print(f"Entrenando {agent_type.upper()}")
        print(f"{'='*60}")

        agent, rewards, losses = train_agent(agent_type, n_episodes)
        results[agent_type] = {
            'agent': agent,
            'rewards': rewards,
            'losses': losses
        }

    # Visualizar comparación
    plt.figure(figsize=(15, 5))

    # Subplot 1: Recompensas
    plt.subplot(1, 2, 1)
    for agent_type, data in results.items():
        rewards = data['rewards']
        # Moving average
        window = 20
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=agent_type.upper(), linewidth=2)

    plt.xlabel('Episodio')
    plt.ylabel('Recompensa (promedio móvil 20)')
    plt.title('Comparación de Algoritmos - Recompensas')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Loss
    plt.subplot(1, 2, 2)
    for agent_type, data in results.items():
        if data['losses']:
            losses = data['losses']
            window = 10
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, label=agent_type.upper(), linewidth=2)

    plt.xlabel('Episodio')
    plt.ylabel('Loss (promedio móvil 10)')
    plt.title('Comparación de Algoritmos - Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lunar_lander_comparison.png', dpi=150)
    print("\n✓ Gráficos guardados en 'lunar_lander_comparison.png'")
    plt.show()

    # Tabla de resultados
    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    print(f"{'Algoritmo':<20} {'Episodios':<12} {'Recompensa Final':<20}")
    print("-"*60)

    for agent_type, data in results.items():
        rewards = data['rewards']
        final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards[-20:])
        print(f"{agent_type.upper():<20} {len(rewards):<12} {final_avg:<20.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Entrenar agente en LunarLander')
    parser.add_argument('--agent', type=str, default='dueling_dqn',
                        choices=['dqn', 'double_dqn', 'dueling_dqn'],
                        help='Tipo de agente a entrenar')
    parser.add_argument('--episodes', type=int, default=600,
                        help='Número de episodios')
    parser.add_argument('--compare', action='store_true',
                        help='Comparar todos los agentes')
    parser.add_argument('--render', action='store_true',
                        help='Renderizar durante entrenamiento')

    args = parser.parse_args()

    if args.compare:
        compare_agents()
    else:
        # Entrenar un solo agente
        agent, rewards, losses = train_agent(
            agent_type=args.agent,
            n_episodes=args.episodes,
            render=args.render
        )

        # Guardar agente
        agent.save(f'lunar_lander_{args.agent}.pth')
        print(f"\n✓ Agente guardado en 'lunar_lander_{args.agent}.pth'")

        # Evaluar
        print("\n" + "="*60)
        print("EVALUACIÓN")
        print("="*60)
        avg_reward = evaluate_agent(agent, n_episodes=10, render=False)

        # Visualizar resultados
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(rewards, alpha=0.3, label='Raw')
        window = 20
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, linewidth=2, label=f'MA-{window}')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.title(f'Entrenamiento - {args.agent.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if losses:
            plt.subplot(1, 2, 2)
            plt.plot(losses, alpha=0.6)
            plt.xlabel('Episodio')
            plt.ylabel('Loss')
            plt.title('Loss durante entrenamiento')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'lunar_lander_{args.agent}.png', dpi=150)
        print(f"✓ Gráficos guardados en 'lunar_lander_{args.agent}.png'")
        plt.show()
