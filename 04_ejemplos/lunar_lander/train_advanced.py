"""
Lunar Lander con Algoritmos Avanzados
======================================

Ejemplo de entrenamiento con PPO y comparación con DQN.
PPO suele converger más rápido y de forma más estable.

Autor: MARK-126
"""

import sys
sys.path.insert(0, '/home/user/Reinforcement-learning-guide')

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from deep_rl.advanced import PPOAgent
from deep_rl.dqn import DuelingDQNAgent


def train_ppo(n_episodes: int = 300) -> Tuple[PPOAgent, List[float]]:
    """
    Entrena un agente PPO en LunarLander-v2.

    PPO es generalmente más sample-efficient y estable que DQN
    en este tipo de ambientes.

    Parámetros:
    -----------
    n_episodes : int
        Número de episodios de entrenamiento

    Retorna:
    --------
    agent : PPOAgent
        Agente entrenado
    rewards : List[float]
        Recompensas por episodio
    """
    env = gym.make('LunarLander-v2')
    state_dim = 8
    action_dim = 4

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=False,  # Acciones discretas
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64
    )

    print(f"\n{'='*60}")
    print(f"Entrenando PPO en LunarLander-v2")
    print(f"{'='*60}\n")

    rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        # Recolectar episodio completo
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_log_probs = []

        while not done:
            action, log_prob = agent.get_action(state, return_log_prob=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_log_probs.append(log_prob)

            state = next_state
            total_reward += reward
            steps += 1

        # Actualizar política con el episodio completo
        agent.update(
            episode_states,
            episode_actions,
            episode_rewards,
            episode_dones,
            episode_log_probs
        )

        rewards.append(total_reward)

        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"Episodio {episode+1}/{n_episodes} | "
                  f"Recompensa: {total_reward:.2f} | "
                  f"Promedio (10): {avg_reward:.2f} | "
                  f"Steps: {steps}")

        # Criterio de éxito
        if len(rewards) >= 100 and np.mean(rewards[-100:]) >= 200:
            print(f"\n✓ ¡Problema resuelto en {episode+1} episodios!")
            print(f"  Promedio de últimos 100 episodios: {np.mean(rewards[-100:]):.2f}")
            break

    env.close()
    return agent, rewards


def compare_ppo_vs_dqn():
    """
    Compara PPO vs Dueling DQN en LunarLander.
    """
    print("\n" + "="*60)
    print("COMPARACIÓN: PPO vs Dueling DQN")
    print("="*60)

    # Entrenar PPO
    print("\n[1/2] Entrenando PPO...")
    ppo_agent, ppo_rewards = train_ppo(n_episodes=300)

    # Entrenar DQN
    print("\n[2/2] Entrenando Dueling DQN...")
    env = gym.make('LunarLander-v2')
    dqn_agent = DuelingDQNAgent(
        state_dim=8,
        action_dim=4,
        use_double_dqn=True,
        learning_rate=5e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    dqn_rewards = []
    for episode in range(300):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = dqn_agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            dqn_agent.store_transition(state, action, reward, next_state, done)

            if len(dqn_agent.replay_buffer) > dqn_agent.batch_size:
                dqn_agent.train_step()

            state = next_state
            total_reward += reward

        dqn_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(dqn_rewards[-10:])
            print(f"Episodio {episode+1}/300 | "
                  f"Recompensa: {total_reward:.2f} | "
                  f"Promedio (10): {avg_reward:.2f}")

        if len(dqn_rewards) >= 100 and np.mean(dqn_rewards[-100:]) >= 200:
            print(f"\n✓ ¡Problema resuelto en {episode+1} episodios!")
            break

    env.close()

    # Visualizar comparación
    plt.figure(figsize=(15, 5))

    # Subplot 1: Recompensas raw
    plt.subplot(1, 3, 1)
    plt.plot(ppo_rewards, alpha=0.3, color='blue', label='PPO')
    plt.plot(dqn_rewards, alpha=0.3, color='red', label='Dueling DQN')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.title('Recompensas por Episodio')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Moving average
    plt.subplot(1, 3, 2)
    window = 20
    ppo_smoothed = np.convolve(ppo_rewards, np.ones(window)/window, mode='valid')
    dqn_smoothed = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
    plt.plot(ppo_smoothed, linewidth=2, color='blue', label='PPO')
    plt.plot(dqn_smoothed, linewidth=2, color='red', label='Dueling DQN')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa (MA-20)')
    plt.title('Recompensas Suavizadas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=200, color='green', linestyle='--', alpha=0.5, label='Objetivo')

    # Subplot 3: Convergencia acumulada
    plt.subplot(1, 3, 3)
    max_len = max(len(ppo_rewards), len(dqn_rewards))
    ppo_cum = np.cumsum(ppo_rewards) / np.arange(1, len(ppo_rewards)+1)
    dqn_cum = np.cumsum(dqn_rewards) / np.arange(1, len(dqn_rewards)+1)
    plt.plot(ppo_cum, linewidth=2, color='blue', label='PPO')
    plt.plot(dqn_cum, linewidth=2, color='red', label='Dueling DQN')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Promedio Acumulada')
    plt.title('Convergencia')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ppo_vs_dqn_comparison.png', dpi=150)
    print("\n✓ Gráficos guardados en 'ppo_vs_dqn_comparison.png'")
    plt.show()

    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    print(f"\nPPO:")
    print(f"  Episodios: {len(ppo_rewards)}")
    print(f"  Recompensa final (últimos 100): {np.mean(ppo_rewards[-100:]) if len(ppo_rewards)>=100 else np.mean(ppo_rewards[-20:]):.2f}")
    print(f"  Mejor episodio: {max(ppo_rewards):.2f}")

    print(f"\nDueling DQN:")
    print(f"  Episodios: {len(dqn_rewards)}")
    print(f"  Recompensa final (últimos 100): {np.mean(dqn_rewards[-100:]) if len(dqn_rewards)>=100 else np.mean(dqn_rewards[-20:]):.2f}")
    print(f"  Mejor episodio: {max(dqn_rewards):.2f}")

    # Análisis
    print(f"\n{'='*60}")
    print("ANÁLISIS")
    print("="*60)

    ppo_avg = np.mean(ppo_rewards[-100:]) if len(ppo_rewards) >= 100 else np.mean(ppo_rewards)
    dqn_avg = np.mean(dqn_rewards[-100:]) if len(dqn_rewards) >= 100 else np.mean(dqn_rewards)

    if ppo_avg > dqn_avg:
        diff = ((ppo_avg - dqn_avg) / dqn_avg) * 100
        print(f"✓ PPO superó a DQN por {diff:.1f}%")
    else:
        diff = ((dqn_avg - ppo_avg) / ppo_avg) * 100
        print(f"✓ DQN superó a PPO por {diff:.1f}%")

    print(f"\nVentajas de PPO:")
    print(f"  - On-policy: más estable y predecible")
    print(f"  - Sample-efficient con experiencia reciente")
    print(f"  - Clipping evita actualizaciones destructivas")

    print(f"\nVentajas de DQN:")
    print(f"  - Off-policy: puede reutilizar experiencia antigua")
    print(f"  - Replay buffer permite más actualizaciones por muestra")
    print(f"  - Mejor para ambientes deterministas")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Entrenar PPO en LunarLander')
    parser.add_argument('--episodes', type=int, default=300,
                        help='Número de episodios')
    parser.add_argument('--compare', action='store_true',
                        help='Comparar PPO vs DQN')

    args = parser.parse_args()

    if args.compare:
        compare_ppo_vs_dqn()
    else:
        # Entrenar solo PPO
        agent, rewards = train_ppo(n_episodes=args.episodes)

        # Guardar
        agent.save('lunar_lander_ppo.pth')
        print("\n✓ Agente guardado en 'lunar_lander_ppo.pth'")

        # Visualizar
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(rewards, alpha=0.3, label='Raw')
        window = 20
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, linewidth=2, label=f'MA-{window}')
        plt.axhline(y=200, color='green', linestyle='--', alpha=0.5, label='Objetivo')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.title('Entrenamiento PPO - LunarLander')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        cum_avg = np.cumsum(rewards) / np.arange(1, len(rewards)+1)
        plt.plot(cum_avg, linewidth=2, color='orange')
        plt.axhline(y=200, color='green', linestyle='--', alpha=0.5, label='Objetivo')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa Promedio')
        plt.title('Convergencia')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('lunar_lander_ppo.png', dpi=150)
        print("✓ Gráficos guardados en 'lunar_lander_ppo.png'")
        plt.show()
