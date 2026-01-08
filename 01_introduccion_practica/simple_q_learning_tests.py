"""
Tests para simple_q_learning.ipynb

Este archivo contiene tests automáticos para validar las implementaciones
de los ejercicios del notebook Simple Q-Learning.
"""

import numpy as np
import gymnasium as gym


def crear_q_table_test(crear_q_table):
    """Test para ejercicio 1: crear_q_table"""
    print("Testing crear_q_table...")
    
    # Test 1: Forma correcta
    Q = crear_q_table(16, 4)
    assert Q.shape == (16, 4), f"❌ Forma incorrecta. Esperado: (16, 4), Obtenido: {Q.shape}"
    
    # Test 2: Tipo correcto
    assert isinstance(Q, np.ndarray), f"❌ Tipo incorrecto. Debe ser numpy.ndarray"
    
    # Test 3: Valores en cero
    assert np.all(Q == 0), "❌ La Q-table debe estar inicializada en ceros"
    
    print("✅ Test pasado! Tu Q-table se creó correctamente.")
    return True


def elegir_accion_test(elegir_accion):
    """Test para ejercicio 2: elegir_accion"""
    print("Testing elegir_accion...")
    
    Q_test = np.array([[0.1, 0.5, 0.2, 0.3],
                       [0.8, 0.2, 0.3, 0.1]])
    
    # Test 1: Epsilon = 0 (siempre explotar)
    for _ in range(10):
        accion = elegir_accion(Q_test, estado=0, epsilon=0.0)
        assert accion == 1, f"❌ Con epsilon=0, debe elegir acción 1 (mejor), obtenido: {accion}"
    
    # Test 2: Epsilon = 1 (siempre explorar)
    acciones = [elegir_accion(Q_test, estado=0, epsilon=1.0) for _ in range(50)]
    acciones_unicas = len(set(acciones))
    assert acciones_unicas >= 2, "❌ Con epsilon=1, debe generar acciones variadas"
    
    # Test 3: Tipo correcto
    accion = elegir_accion(Q_test, estado=0, epsilon=0.5)
    assert isinstance(accion, (int, np.integer)), f"❌ Debe retornar int, obtenido: {type(accion)}"
    
    # Test 4: Rango válido
    assert 0 <= accion < 4, f"❌ Acción debe estar entre 0 y 3, obtenido: {accion}"
    
    print("✅ Test pasado! Eliges acciones correctamente con epsilon-greedy.")
    return True


def actualizar_q_test(actualizar_q):
    """Test para ejercicio 3: actualizar_q"""
    print("Testing actualizar_q...")
    
    # Test 1: Actualización sin terminar
    Q_test = np.zeros((3, 2))
    Q_test[1] = [0.5, 0.3]
    
    Q_test = actualizar_q(Q_test, estado=0, accion=0, recompensa=1.0,
                         siguiente_estado=1, done=False, alpha=0.1, gamma=0.9)
    
    # target = 1.0 + 0.9 * 0.5 = 1.45
    # nuevo_q = 0 + 0.1 * 1.45 = 0.145
    esperado = 0.145
    assert np.isclose(Q_test[0, 0], esperado, atol=0.001), \
        f"❌ Q[0,0] incorrecto. Esperado: {esperado}, Obtenido: {Q_test[0, 0]}"
    
    # Test 2: Actualización con done=True
    Q_test2 = np.zeros((3, 2))
    Q_test2[1] = [0.5, 0.3]
    
    Q_test2 = actualizar_q(Q_test2, estado=0, accion=0, recompensa=1.0,
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


def entrenar_q_learning_test(entrenar_q_learning):
    """Test para ejercicio 4: entrenar_q_learning"""
    print("Testing entrenar_q_learning...")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    # Test 1: Retorna Q-table y recompensas
    Q, recompensas = entrenar_q_learning(env, n_episodios=100, alpha=0.1, 
                                         gamma=0.99, epsilon=0.1)
    
    assert isinstance(Q, np.ndarray), "❌ Q debe ser numpy array"
    assert isinstance(recompensas, list), "❌ recompensas debe ser lista"
    
    # Test 2: Forma correcta de Q
    assert Q.shape == (16, 4), f"❌ Q debe tener forma (16, 4), obtenido: {Q.shape}"
    
    # Test 3: Número correcto de episodios
    assert len(recompensas) == 100, f"❌ Debe haber 100 recompensas, obtenido: {len(recompensas)}"
    
    # Test 4: Q-values cambiaron (aprendió algo)
    assert not np.all(Q == 0), "❌ Q-table debe haber cambiado durante entrenamiento"
    
    # Test 5: Recompensas válidas
    assert all(r in [0.0, 1.0, 0, 1] for r in recompensas), \
        "❌ Recompensas deben ser 0 o 1 en FrozenLake"
    
    env.close()
    print("✅ Test pasado! Tu agente Q-Learning funciona correctamente.")
    print(f"   Nota: Con más episodios (2000+) el agente alcanza ~95% de éxito.")
    return True


if __name__ == "__main__":
    print("Ejecutando todos los tests...")
    print("=" * 50)
    
    print("\nPara ejecutar los tests, importa las funciones desde el notebook")
    print("y llama a cada función de test.")
