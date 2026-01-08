"""
Tests para gym_basics.ipynb

Este archivo contiene tests automáticos para validar las implementaciones
de los ejercicios del notebook Gymnasium Basics.
"""

import gymnasium as gym
import numpy as np


def explorar_ambiente_test(explorar_ambiente):
    """Test para ejercicio 1: explorar_ambiente"""
    print("Testing explorar_ambiente...")
    
    # Test 1: FrozenLake
    n_estados, n_acciones = explorar_ambiente('FrozenLake-v1')
    assert n_estados == 16, f"❌ FrozenLake debe tener 16 estados, obtenido: {n_estados}"
    assert n_acciones == 4, f"❌ FrozenLake debe tener 4 acciones, obtenido: {n_acciones}"
    
    # Test 2: Tipos correctos
    assert isinstance(n_estados, (int, np.integer)), f"❌ n_estados debe ser int, obtenido: {type(n_estados)}"
    assert isinstance(n_acciones, (int, np.integer)), f"❌ n_acciones debe ser int, obtenido: {type(n_acciones)}"
    
    print("✅ Test pasado! Exploras ambientes correctamente.")
    return True


def resetear_ambiente_test(resetear_ambiente):
    """Test para ejercicio 2: resetear_ambiente"""
    print("Testing resetear_ambiente...")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    # Test 1: Estado inicial correcto
    estado = resetear_ambiente(env)
    assert estado == 0, f"❌ Estado inicial debe ser 0, obtenido: {estado}"
    
    # Test 2: Tipo correcto
    assert isinstance(estado, (int, np.integer)), f"❌ Estado debe ser int, obtenido: {type(estado)}"
    
    # Test 3: Reset múltiple
    estado2 = resetear_ambiente(env)
    assert estado2 == 0, "❌ Reset debe retornar siempre el estado inicial"
    
    env.close()
    print("✅ Test pasado! Reseteas ambientes correctamente.")
    return True


def tomar_accion_test(tomar_accion):
    """Test para ejercicio 3: tomar_accion"""
    print("Testing tomar_accion...")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    env.reset()
    
    # Test 1: Acción derecha desde estado 0
    siguiente_estado, recompensa, done = tomar_accion(env, 2)
    assert siguiente_estado == 1, f"❌ Desde estado 0, acción 2 (derecha) debe ir a estado 1, obtenido: {siguiente_estado}"
    assert recompensa == 0.0, f"❌ Recompensa debe ser 0.0, obtenido: {recompensa}"
    assert done == False, f"❌ done debe ser False, obtenido: {done}"
    
    # Test 2: Tipos correctos
    assert isinstance(siguiente_estado, (int, np.integer)), "❌ siguiente_estado debe ser int"
    assert isinstance(recompensa, (int, float, np.number)), "❌ recompensa debe ser numérico"
    assert isinstance(done, bool), "❌ done debe ser bool"
    
    env.close()
    print("✅ Test pasado! Tomas acciones correctamente.")
    return True


def ejecutar_episodio_random_test(ejecutar_episodio_random):
    """Test para ejercicio 4: ejecutar_episodio_random"""
    print("Testing ejecutar_episodio_random...")
    
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    # Test 1: Retorna lista
    recompensas = ejecutar_episodio_random(env)
    assert isinstance(recompensas, list), f"❌ Debe retornar una lista, obtenido: {type(recompensas)}"
    
    # Test 2: Lista no vacía
    assert len(recompensas) > 0, "❌ La lista de recompensas no debe estar vacía"
    
    # Test 3: Todas las recompensas son números
    assert all(isinstance(r, (int, float, np.number)) for r in recompensas), \
        "❌ Todas las recompensas deben ser números"
    
    # Test 4: Recompensas válidas (0 o 1 en FrozenLake)
    assert all(r in [0.0, 1.0, 0, 1] for r in recompensas), \
        "❌ Recompensas deben ser 0 o 1 en FrozenLake"
    
    # Test 5: Ejecutar múltiples episodios
    for _ in range(5):
        recompensas = ejecutar_episodio_random(env)
        assert len(recompensas) > 0, "❌ Cada episodio debe tener al menos un paso"
    
    env.close()
    print("✅ Test pasado! Ejecutas episodios correctamente.")
    return True


def calcular_recompensa_total_test(calcular_recompensa_total):
    """Test para ejercicio 5: calcular_recompensa_total"""
    print("Testing calcular_recompensa_total...")
    
    # Test 1: Lista simple
    recompensas1 = [0, 0, 0, 1]
    total1 = calcular_recompensa_total(recompensas1)
    assert total1 == 1, f"❌ Esperado: 1, obtenido: {total1}"
    
    # Test 2: Todas ceros
    recompensas2 = [0, 0, 0]
    total2 = calcular_recompensa_total(recompensas2)
    assert total2 == 0, f"❌ Esperado: 0, obtenido: {total2}"
    
    # Test 3: Múltiples recompensas
    recompensas3 = [1, 1, 1]
    total3 = calcular_recompensa_total(recompensas3)
    assert total3 == 3, f"❌ Esperado: 3, obtenido: {total3}"
    
    # Test 4: Tipo correcto
    assert isinstance(total1, (int, float, np.number)), \
        f"❌ Debe retornar un número, obtenido: {type(total1)}"
    
    print("✅ Test pasado! Calculas recompensa total correctamente.")
    return True


if __name__ == "__main__":
    print("Ejecutando todos los tests...")
    print("=" * 50)
    
    print("\nPara ejecutar los tests, importa las funciones desde el notebook")
    print("y llama a cada función de test.")
