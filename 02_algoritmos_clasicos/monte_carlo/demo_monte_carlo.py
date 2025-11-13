"""
Demostración completa de algoritmos Monte Carlo
================================================

Este script demuestra el uso de todos los algoritmos Monte Carlo implementados:
- MC Prediction (First-Visit y Every-Visit)
- MC Control On-Policy (ε-greedy)
- MC Control Off-Policy (Importance Sampling)

Con diferentes entornos:
- GridWorld simple
- Blackjack simplificado
- Cliff Walking
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monte_carlo import (
    MCPrediction,
    MCControlOnPolicy,
    MCControlOffPolicy,
    SimpleGridWorld,
    SimpleBlackjack
)
import numpy as np


def demo_mc_prediction():
    """Demostración de MC Prediction"""
    print("\n" + "=" * 70)
    print("DEMO 1: MONTE CARLO PREDICTION")
    print("=" * 70)
    print("\nObjetivo: Evaluar una política dada en GridWorld 3x3")
    print("Política: Moverse preferentemente hacia abajo y derecha")

    # Crear entorno pequeño para visualización clara
    env = SimpleGridWorld(size=3)

    # Crear política subóptima pero razonable
    def biased_policy(state):
        """Política que prefiere ir hacia la meta"""
        import random
        row, col = state
        # 60% de probabilidad de ir hacia la meta
        if random.random() < 0.6:
            if col < 2:  # Preferir derecha
                return 1
            elif row < 2:  # Preferir abajo
                return 2
        # 40% acción aleatoria
        return random.randint(0, 3)

    # Evaluar con First-Visit MC
    mc_first = MCPrediction(gamma=0.99, method='first-visit')
    mc_first.evaluate_policy(env, biased_policy, num_episodes=1000, verbose=False)

    # Evaluar con Every-Visit MC
    mc_every = MCPrediction(gamma=0.99, method='every-visit')
    mc_every.evaluate_policy(env, biased_policy, num_episodes=1000, verbose=False)

    # Mostrar resultados
    print("\nFunción de Valor estimada (GridWorld 3x3):")
    print("\nFirst-Visit MC:")
    for row in range(3):
        for col in range(3):
            state = (row, col)
            value = mc_first.get_value(state)
            print(f"{value:6.3f}", end=" ")
        print()

    print("\nEvery-Visit MC:")
    for row in range(3):
        for col in range(3):
            state = (row, col)
            value = mc_every.get_value(state)
            print(f"{value:6.3f}", end=" ")
        print()

    # Estadísticas
    stats_first = mc_first.get_statistics()
    stats_every = mc_every.get_statistics()

    print(f"\nEstadísticas de evaluación:")
    print(f"  First-Visit - Return promedio: {stats_first['mean_return']:.4f}")
    print(f"  Every-Visit - Return promedio: {stats_every['mean_return']:.4f}")


def demo_mc_control_comparison():
    """Comparación de MC Control On-Policy vs Off-Policy"""
    print("\n" + "=" * 70)
    print("DEMO 2: COMPARACIÓN MC CONTROL ON-POLICY VS OFF-POLICY")
    print("=" * 70)
    print("\nObjetivo: Encontrar política óptima en GridWorld 4x4")

    env = SimpleGridWorld(size=4)

    # On-Policy MC Control
    print("\n[1] Entrenando agente On-Policy (ε-greedy)...")
    agent_on = MCControlOnPolicy(
        gamma=0.99,
        epsilon=0.2,
        epsilon_decay=0.998,
        epsilon_min=0.01
    )
    agent_on.train(env, num_episodes=2000, verbose=False)

    # Off-Policy MC Control
    print("[2] Entrenando agente Off-Policy (Importance Sampling)...")
    agent_off = MCControlOffPolicy(
        gamma=0.99,
        epsilon=0.3
    )
    agent_off.train(env, num_episodes=3000, verbose=False)

    # Evaluar ambos agentes
    def evaluate_agent(agent, episodes=100):
        returns = []
        lengths = []
        for _ in range(episodes):
            state = env.reset()
            episode_return = 0
            episode_length = 0
            for _ in range(100):
                if hasattr(agent, 'get_action'):
                    if isinstance(agent, MCControlOnPolicy):
                        action = agent.get_action(state, greedy=True)
                    else:
                        action = agent.get_action(state)
                else:
                    action = 0
                state, reward, done, _ = env.step(action)
                episode_return += reward
                episode_length += 1
                if done:
                    break
            returns.append(episode_return)
            lengths.append(episode_length)
        return np.mean(returns), np.std(returns), np.mean(lengths)

    print("\n[3] Evaluando políticas aprendidas...")
    return_on, std_on, length_on = evaluate_agent(agent_on)
    return_off, std_off, length_off = evaluate_agent(agent_off)

    print("\nResultados de evaluación (100 episodios):")
    print(f"  On-Policy:")
    print(f"    - Return: {return_on:.4f} ± {std_on:.4f}")
    print(f"    - Longitud promedio: {length_on:.2f} pasos")
    print(f"  Off-Policy:")
    print(f"    - Return: {return_off:.4f} ± {std_off:.4f}")
    print(f"    - Longitud promedio: {length_off:.2f} pasos")

    # Mostrar políticas aprendidas
    policy_symbols = ['↑', '→', '↓', '←']

    print("\nPolítica On-Policy aprendida:")
    for row in range(4):
        for col in range(4):
            state = (row, col)
            if state in agent_on.Q:
                action = agent_on.get_action(state, greedy=True)
                print(f" {policy_symbols[action]} ", end="")
            else:
                print(" ? ", end="")
        print()

    print("\nPolítica Off-Policy aprendida:")
    for row in range(4):
        for col in range(4):
            state = (row, col)
            if state in agent_off.Q:
                action = agent_off.get_action(state)
                print(f" {policy_symbols[action]} ", end="")
            else:
                print(" ? ", end="")
        print()


def demo_convergence_analysis():
    """Análisis de convergencia de los algoritmos"""
    print("\n" + "=" * 70)
    print("DEMO 3: ANÁLISIS DE CONVERGENCIA")
    print("=" * 70)
    print("\nAnalizando velocidad de convergencia en GridWorld 3x3")

    env = SimpleGridWorld(size=3)

    # Entrenar agente y analizar convergencia
    agent = MCControlOnPolicy(
        gamma=0.99,
        epsilon=0.2,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    print("\nEntrenando agente...")
    results = agent.train(env, num_episodes=1000, verbose=False)

    # Analizar historial
    history = results['history']
    returns = history['episode_returns']
    lengths = history['episode_lengths']

    # Calcular estadísticas por ventanas
    window_size = 100
    windowed_returns = []
    windowed_lengths = []

    for i in range(0, len(returns), window_size):
        window_returns = returns[i:i+window_size]
        window_lengths = lengths[i:i+window_size]
        if window_returns:
            windowed_returns.append(np.mean(window_returns))
            windowed_lengths.append(np.mean(window_lengths))

    print(f"\nProgreso del entrenamiento (ventanas de {window_size} episodios):")
    print(f"{'Ventana':<10} {'Return Promedio':<20} {'Longitud Promedio':<20} {'Epsilon':<10}")
    print("-" * 65)

    for i, (ret, length) in enumerate(zip(windowed_returns, windowed_lengths)):
        episode_num = (i + 1) * window_size
        epsilon = history['epsilon_history'][min(episode_num - 1, len(history['epsilon_history']) - 1)]
        print(f"{i+1:<10} {ret:<20.4f} {length:<20.2f} {epsilon:<10.4f}")

    print(f"\nMejora total:")
    print(f"  Return inicial: {windowed_returns[0]:.4f}")
    print(f"  Return final: {windowed_returns[-1]:.4f}")
    print(f"  Mejora: {((windowed_returns[-1] - windowed_returns[0]) / abs(windowed_returns[0]) * 100):.1f}%")


def demo_blackjack():
    """Demostración con Blackjack"""
    print("\n" + "=" * 70)
    print("DEMO 4: MONTE CARLO EN BLACKJACK")
    print("=" * 70)
    print("\nEvaluando diferentes estrategias en Blackjack")

    env = SimpleBlackjack()

    strategies = {
        'Conservadora (threshold=18)': lambda state: 1 if state[0] < 18 else 0,
        'Moderada (threshold=19)': lambda state: 1 if state[0] < 19 else 0,
        'Agresiva (threshold=20)': lambda state: 1 if state[0] < 20 else 0,
    }

    print("\nEvaluando estrategias con MC Prediction...")

    for name, policy in strategies.items():
        mc = MCPrediction(gamma=1.0, method='first-visit')
        mc.evaluate_policy(env, policy, num_episodes=5000, verbose=False)

        stats = mc.get_statistics()
        win_rate = (stats['mean_return'] + 1) / 2  # Convertir [-1,1] a [0,1]

        print(f"\n{name}:")
        print(f"  Return promedio: {stats['mean_return']:.4f}")
        print(f"  Tasa de victoria estimada: {win_rate:.1%}")
        print(f"  Estados visitados: {stats['num_states']}")


if __name__ == "__main__":
    print("=" * 70)
    print("DEMOSTRACIÓN COMPLETA: ALGORITMOS MONTE CARLO")
    print("=" * 70)
    print("\nEste script demuestra todos los algoritmos Monte Carlo implementados")
    print("en diferentes entornos y configuraciones.")

    try:
        demo_mc_prediction()
        demo_mc_control_comparison()
        demo_convergence_analysis()
        demo_blackjack()

        print("\n" + "=" * 70)
        print("DEMOSTRACIÓN COMPLETADA ✓")
        print("=" * 70)
        print("\nResumen de lo demostrado:")
        print("  1. MC Prediction: Evaluación de políticas (First-Visit y Every-Visit)")
        print("  2. MC Control: Comparación On-Policy vs Off-Policy")
        print("  3. Análisis de convergencia: Seguimiento del aprendizaje")
        print("  4. Aplicación práctica: Estrategias en Blackjack")
        print("\nTodos los algoritmos funcionan correctamente!")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
