"""
Script de prueba para verificar los módulos Monte Carlo
"""

import sys
import os

# Agregar el directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monte_carlo import (
    MCPrediction,
    MCControlOnPolicy,
    MCControlOffPolicy,
    SimpleGridWorld,
    SimpleBlackjack,
    CliffWalking
)

def test_mc_prediction():
    """Prueba rápida de MC Prediction"""
    print("Probando MCPrediction...")
    env = SimpleGridWorld(size=3)
    mc = MCPrediction(gamma=0.99, method='first-visit')

    def random_policy(state):
        import random
        return random.randint(0, 3)

    results = mc.evaluate_policy(env, random_policy, num_episodes=100, verbose=False)
    assert len(mc.V) > 0, "No se estimaron valores"
    print("  ✓ MCPrediction funciona correctamente")

def test_mc_control_on_policy():
    """Prueba rápida de MC Control On-Policy"""
    print("Probando MCControlOnPolicy...")
    env = SimpleGridWorld(size=3)
    agent = MCControlOnPolicy(gamma=0.99, epsilon=0.1)

    results = agent.train(env, num_episodes=100, verbose=False)
    assert len(agent.Q) > 0, "No se estimaron Q values"
    print("  ✓ MCControlOnPolicy funciona correctamente")

def test_mc_control_off_policy():
    """Prueba rápida de MC Control Off-Policy"""
    print("Probando MCControlOffPolicy...")
    env = SimpleGridWorld(size=3)
    agent = MCControlOffPolicy(gamma=0.99, epsilon=0.2)

    results = agent.train(env, num_episodes=100, verbose=False)
    assert len(agent.Q) > 0, "No se estimaron Q values"
    print("  ✓ MCControlOffPolicy funciona correctamente")

def test_environments():
    """Prueba los entornos"""
    print("Probando entornos...")

    # GridWorld
    env1 = SimpleGridWorld(size=3)
    state = env1.reset()
    state, reward, done, info = env1.step(1)
    print("  ✓ SimpleGridWorld funciona")

    # Blackjack
    env2 = SimpleBlackjack()
    state = env2.reset()
    state, reward, done, info = env2.step(0)
    print("  ✓ SimpleBlackjack funciona")

    # Cliff Walking
    env3 = CliffWalking()
    state = env3.reset()
    state, reward, done, info = env3.step(1)
    print("  ✓ CliffWalking funciona")

if __name__ == "__main__":
    print("=" * 60)
    print("TEST SUITE: Algoritmos Monte Carlo")
    print("=" * 60)

    try:
        test_environments()
        test_mc_prediction()
        test_mc_control_on_policy()
        test_mc_control_off_policy()

        print("\n" + "=" * 60)
        print("TODOS LOS TESTS PASARON ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
