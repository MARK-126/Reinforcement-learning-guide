"""
Tests para SARSA_Exercise.ipynb
"""
import numpy as np
import gymnasium as gym


def sarsa_update_test(sarsa_update):
    """Test para ejercicio 1: sarsa_update"""
    print("Testing sarsa_update...")
    
    # Test 1: Actualización sin terminar
    Q_test = np.zeros((3, 2))
    Q_test[1] = [0.5, 0.3]
    
    Q_test, td_error = sarsa_update(Q_test, state=0, action=0, reward=1.0,
                                    next_state=1, next_action=1, done=False, 
                                    alpha=0.1, gamma=0.9)
    
    # target = 1.0 + 0.9 * 0.3 = 1.27 (usa next_action=1, no max!)
    # nuevo_q = 0 + 0.1 * 1.27 = 0.127
    esperado = 0.127
    assert np.isclose(Q_test[0, 0], esperado, atol=0.001), \
        f"❌ Q[0,0] incorrecto. Esperado: {esperado}, Obtenido: {Q_test[0, 0]}"
    
    # Test 2: Actualización con done=True
    Q_test2 = np.zeros((3, 2))
    Q_test2, td_error2 = sarsa_update(Q_test2, state=0, action=0, reward=1.0,
                                      next_state=1, next_action=1, done=True, 
                                      alpha=0.1, gamma=0.9)
    
    esperado2 = 0.1
    assert np.isclose(Q_test2[0, 0], esperado2, atol=0.001), \
        f"❌ Q[0,0] con done=True incorrecto. Esperado: {esperado2}, Obtenido: {Q_test2[0, 0]}"
    
    print("✅ Test pasado! SARSA update funciona correctamente.")
    return True


def run_single_episode_sarsa_test(run_single_episode_sarsa):
    """Test para ejercicio 2a: run_single_episode_sarsa"""
    print("Testing run_single_episode_sarsa...")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    Q = np.zeros((16, 4))
    
    # Test 1: Retorna Q y recompensa
    Q_new, recompensa_total = run_single_episode_sarsa(env, Q, epsilon=0.1, alpha=0.1, gamma=0.99)
    
    assert isinstance(Q_new, np.ndarray), "❌ Debe retornar Q-table"
    assert isinstance(recompensa_total, (int, float, np.number)), "❌ Debe retornar recompensa numérica"
    
    # Test 2: Q-table tiene la forma correcta
    assert Q_new.shape == (16, 4), f"❌ Q debe tener forma (16, 4), obtenido: {Q_new.shape}"
    
    env.close()
    print("✅ Test pasado! Ejecutas episodios SARSA correctamente.")
    return True


def train_sarsa_simple_test(train_sarsa_simple):
    """Test para ejercicio 2b: train_sarsa_simple"""
    print("Testing train_sarsa_simple...")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    # Test 1: Retorna Q y recompensas
    Q, recompensas = train_sarsa_simple(env, n_episodes=100, alpha=0.1, gamma=0.99, epsilon=0.1)
    
    assert isinstance(Q, np.ndarray), "❌ Debe retornar Q-table"
    assert isinstance(recompensas, list), "❌ Debe retornar lista de recompensas"
    
    # Test 2: Número correcto de episodios
    assert len(recompensas) == 100, f"❌ Debe haber 100 recompensas, obtenido: {len(recompensas)}"
    
    env.close()
    print("✅ Test pasado! SARSA simple funciona correctamente.")
    return True


def train_sarsa_with_decay_test(train_sarsa_with_decay):
    """Test para ejercicio 2c: train_sarsa_with_decay"""
    print("Testing train_sarsa_with_decay...")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    # Test 1: Retorna Q y recompensas
    Q, recompensas = train_sarsa_with_decay(env, n_episodes=1000, alpha=0.1, gamma=0.99, 
                                            epsilon_start=1.0, epsilon_end=0.01)
    
    assert isinstance(Q, np.ndarray), "❌ Debe retornar Q-table"
    assert len(recompensas) == 1000, f"❌ Debe haber 1000 recompensas"
    
    # Test 2: Aprende
    promedio_inicial = np.mean(recompensas[:100])
    promedio_final = np.mean(recompensas[-100:])
    assert promedio_final >= promedio_inicial, \
        f"❌ Agente debería mejorar. Inicial: {promedio_inicial:.2f}, Final: {promedio_final:.2f}"
    
    env.close()
    print("✅ Test pasado! SARSA con decay funciona correctamente.")
    print(f"   Mejora: {promedio_inicial*100:.0f}% → {promedio_final*100:.0f}%")
    return True


if __name__ == "__main__":
    print("Tests para SARSA_Exercise.ipynb")
    print("=" * 50)
