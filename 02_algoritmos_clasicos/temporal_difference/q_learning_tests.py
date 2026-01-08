"""
Tests para Q_Learning_Exercise.ipynb

Este archivo contiene tests automáticos para validar las implementaciones
de los ejercicios del notebook Q-Learning.
"""

import numpy as np
import gymnasium as gym


def initialize_q_table_test(initialize_q_table):
    """Test para ejercicio 1: initialize_q_table"""
    print("Testing initialize_q_table...")
    
    # Test 1: Forma correcta
    Q = initialize_q_table(16, 4)
    assert Q.shape == (16, 4), f"❌ Forma incorrecta. Esperado: (16, 4), Obtenido: {Q.shape}"
    
    # Test 2: Tipo correcto
    assert isinstance(Q, np.ndarray), f"❌ Tipo incorrecto. Debe ser numpy.ndarray"
    
    # Test 3: Valores en cero
    assert np.all(Q == 0), "❌ La Q-table debe estar inicializada en ceros"
    
    # Test 4: Diferentes tamaños
    Q2 = initialize_q_table(10, 5)
    assert Q2.shape == (10, 5), f"❌ Forma incorrecta para (10, 5)"
    
    print("✅ Test pasado! Tu Q-table se creó correctamente.")
    return True


def epsilon_greedy_action_test(epsilon_greedy_action):
    """Test para ejercicio 2: epsilon_greedy_action"""
    print("Testing epsilon_greedy_action...")
    
    Q_test = np.array([[0.1, 0.5, 0.2, 0.3],
                       [0.8, 0.2, 0.3, 0.1]])
    
    # Test 1: Epsilon = 0 (siempre explotar)
    for _ in range(10):
        accion = epsilon_greedy_action(Q_test, estado=0, epsilon=0.0)
        assert accion == 1, f"❌ Con epsilon=0, debe elegir acción 1 (mejor), obtenido: {accion}"
    
    # Test 2: Epsilon = 1 (siempre explorar)
    acciones = [epsilon_greedy_action(Q_test, estado=0, epsilon=1.0) for _ in range(50)]
    acciones_unicas = len(set(acciones))
    assert acciones_unicas >= 2, "❌ Con epsilon=1, debe generar acciones variadas"
    
    # Test 3: Tipo correcto
    accion = epsilon_greedy_action(Q_test, estado=0, epsilon=0.5)
    assert isinstance(accion, (int, np.integer)), f"❌ Debe retornar int, obtenido: {type(accion)}"
    
    # Test 4: Rango válido
    assert 0 <= accion < 4, f"❌ Acción debe estar entre 0 y 3, obtenido: {accion}"
    
    print("✅ Test pasado! Eliges acciones correctamente con epsilon-greedy.")
    return True


def q_learning_update_test(q_learning_update):
    """Test para ejercicio 3: q_learning_update"""
    print("Testing q_learning_update...")
    
    # Test 1: Actualización sin terminar
    Q_test = np.zeros((3, 2))
    Q_test[1] = [0.5, 0.3]
    
    Q_test = q_learning_update(Q_test, estado=0, accion=0, recompensa=1.0,
                               siguiente_estado=1, done=False, alpha=0.1, gamma=0.9)
    
    # target = 1.0 + 0.9 * 0.5 = 1.45
    # nuevo_q = 0 + 0.1 * 1.45 = 0.145
    esperado = 0.145
    assert np.isclose(Q_test[0, 0], esperado, atol=0.001), \
        f"❌ Q[0,0] incorrecto. Esperado: {esperado}, Obtenido: {Q_test[0, 0]}"
    
    # Test 2: Actualización con done=True
    Q_test2 = np.zeros((3, 2))
    Q_test2[1] = [0.5, 0.3]
    
    Q_test2 = q_learning_update(Q_test2, estado=0, accion=0, recompensa=1.0,
                                siguiente_estado=1, done=True, alpha=0.1, gamma=0.9)
    
    # target = 1.0 (sin futuro)
    # nuevo_q = 0 + 0.1 * 1.0 = 0.1
    esperado2 = 0.1
    assert np.isclose(Q_test2[0, 0], esperado2, atol=0.001), \
        f"❌ Q[0,0] con done=True incorrecto. Esperado: {esperado2}, Obtenido: {Q_test2[0, 0]}"
    
    # Test 3: Otros valores no cambian
    assert Q_test[1, 0] == 0.5, "❌ Otros valores no deben cambiar"
    
    print("✅ Test pasado! Actualizas Q-values correctamente.")
    return True


def run_single_episode_test(run_single_episode):
    """Test para ejercicio 4a: run_single_episode"""
    print("Testing run_single_episode...")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    Q = np.zeros((16, 4))
    
    # Test 1: Retorna Q y recompensa
    Q_new, recompensa_total = run_single_episode(env, Q, epsilon=0.1, alpha=0.1, gamma=0.99)
    
    assert isinstance(Q_new, np.ndarray), "❌ Debe retornar Q-table"
    assert isinstance(recompensa_total, (int, float, np.number)), "❌ Debe retornar recompensa numérica"
    
    # Test 2: Q-table tiene la forma correcta
    assert Q_new.shape == (16, 4), f"❌ Q debe tener forma (16, 4), obtenido: {Q_new.shape}"
    
    # Test 3: Recompensa válida (0 o 1 en FrozenLake)
    assert recompensa_total in [0.0, 1.0, 0, 1], \
        f"❌ Recompensa debe ser 0 o 1, obtenido: {recompensa_total}"
    
    env.close()
    print("✅ Test pasado! Ejecutas episodios correctamente.")
    return True


def train_q_learning_simple_test(train_q_learning_simple):
    """Test para ejercicio 4b: train_q_learning_simple"""
    print("Testing train_q_learning_simple...")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    # Test 1: Retorna Q y recompensas
    Q, recompensas = train_q_learning_simple(env, n_episodes=100, alpha=0.1, gamma=0.99, epsilon=0.1)
    
    assert isinstance(Q, np.ndarray), "❌ Debe retornar Q-table"
    assert isinstance(recompensas, list), "❌ Debe retornar lista de recompensas"
    
    # Test 2: Forma correcta
    assert Q.shape == (16, 4), f"❌ Q debe tener forma (16, 4), obtenido: {Q.shape}"
    
    # Test 3: Número correcto de episodios
    assert len(recompensas) == 100, f"❌ Debe haber 100 recompensas, obtenido: {len(recompensas)}"
    
    # Test 4: Q-values cambiaron
    assert not np.all(Q == 0), "❌ Q-table debe haber cambiado durante entrenamiento"
    
    env.close()
    print("✅ Test pasado! Entrenas correctamente (versión simple).")
    return True


def train_q_learning_with_decay_test(train_q_learning_with_decay):
    """Test para ejercicio 4c: train_q_learning_with_decay"""
    print("Testing train_q_learning_with_decay...")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    # Test 1: Retorna Q y recompensas
    Q, recompensas = train_q_learning_with_decay(env, n_episodes=1000, alpha=0.1, gamma=0.99, 
                                                  epsilon_start=1.0, epsilon_end=0.01)
    
    assert isinstance(Q, np.ndarray), "❌ Debe retornar Q-table"
    assert isinstance(recompensas, list), "❌ Debe retornar lista de recompensas"
    
    # Test 2: Forma correcta
    assert Q.shape == (16, 4), f"❌ Q debe tener forma (16, 4), obtenido: {Q.shape}"
    
    # Test 3: Número correcto de episodios
    assert len(recompensas) == 1000, f"❌ Debe haber 1000 recompensas, obtenido: {len(recompensas)}"
    
    # Test 4: Aprende (últimos episodios mejor que primeros)
    promedio_inicial = np.mean(recompensas[:100])
    promedio_final = np.mean(recompensas[-100:])
    assert promedio_final >= promedio_inicial, \
        f"❌ Agente debería mejorar con el tiempo. Inicial: {promedio_inicial:.2f}, Final: {promedio_final:.2f}"
    
    env.close()
    print("✅ Test pasado! Entrenas correctamente con epsilon decay.")
    print(f"   Mejora: {promedio_inicial*100:.0f}% → {promedio_final*100:.0f}%")
    return True


if __name__ == "__main__":
    print("Ejecutando todos los tests...")
    print("=" * 50)
    
    print("\nPara ejecutar los tests, importa las funciones desde el notebook")
    print("y llama a cada función de test.")
