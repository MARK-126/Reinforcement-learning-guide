"""
Tests para numpy_for_rl.ipynb

Este archivo contiene tests automáticos para validar las implementaciones
de los ejercicios del notebook NumPy para RL.
"""

import numpy as np


def crear_q_table_test(crear_q_table):
    """Test para ejercicio 1: crear_q_table"""
    print("Testing crear_q_table...")
    
    # Test 1: Forma correcta
    Q = crear_q_table(4, 2)
    assert Q.shape == (4, 2), f"❌ Forma incorrecta. Esperado: (4, 2), Obtenido: {Q.shape}"
    
    # Test 2: Tipo correcto
    assert isinstance(Q, np.ndarray), f"❌ Tipo incorrecto. Debe ser numpy.ndarray, obtenido: {type(Q)}"
    
    # Test 3: Valores en cero
    assert np.all(Q == 0), "❌ La Q-table debe estar inicializada en ceros"
    
    # Test 4: Diferentes tamaños
    Q2 = crear_q_table(10, 5)
    assert Q2.shape == (10, 5), f"❌ Forma incorrecta para (10, 5)"
    
    print("✅ Test pasado! Tu Q-table se creó correctamente.")
    return True


def obtener_q_value_test(obtener_q_value):
    """Test para ejercicio 2: obtener_q_value"""
    print("Testing obtener_q_value...")
    
    Q_test = np.array([[0.5, 0.3],
                       [0.2, 0.8],
                       [0.1, 0.4]])
    
    # Test 1: Valor correcto
    valor = obtener_q_value(Q_test, 1, 1)
    assert valor == 0.8, f"❌ Valor incorrecto. Esperado: 0.8, Obtenido: {valor}"
    
    # Test 2: Otro valor
    valor2 = obtener_q_value(Q_test, 0, 0)
    assert valor2 == 0.5, f"❌ Valor incorrecto. Esperado: 0.5, Obtenido: {valor2}"
    
    # Test 3: Tercer valor
    valor3 = obtener_q_value(Q_test, 2, 1)
    assert valor3 == 0.4, f"❌ Valor incorrecto. Esperado: 0.4, Obtenido: {valor3}"
    
    print("✅ Test pasado! Puedes acceder a Q-values correctamente.")
    return True


def seleccionar_mejor_accion_test(seleccionar_mejor_accion):
    """Test para ejercicio 3: seleccionar_mejor_accion"""
    print("Testing seleccionar_mejor_accion...")
    
    Q_test = np.array([[0.5, 0.3],
                       [0.2, 0.8],
                       [0.9, 0.1]])
    
    # Test 1: Estado 0
    accion = seleccionar_mejor_accion(Q_test, 0)
    assert accion == 0, f"❌ Para estado 0, esperado: 0, obtenido: {accion}"
    
    # Test 2: Estado 1
    accion2 = seleccionar_mejor_accion(Q_test, 1)
    assert accion2 == 1, f"❌ Para estado 1, esperado: 1, obtenido: {accion2}"
    
    # Test 3: Estado 2
    accion3 = seleccionar_mejor_accion(Q_test, 2)
    assert accion3 == 0, f"❌ Para estado 2, esperado: 0, obtenido: {accion3}"
    
    print("✅ Test pasado! Seleccionas la mejor acción correctamente.")
    return True


def actualizar_q_value_test(actualizar_q_value):
    """Test para ejercicio 4: actualizar_q_value"""
    print("Testing actualizar_q_value...")
    
    Q_test = np.zeros((3, 2))
    
    # Test 1: Primera actualización
    Q_test = actualizar_q_value(Q_test, 1, 0, 0.5)
    assert Q_test[1, 0] == 0.5, f"❌ Después de +0.5, esperado: 0.5, obtenido: {Q_test[1, 0]}"
    
    # Test 2: Segunda actualización (acumulativa)
    Q_test = actualizar_q_value(Q_test, 1, 0, 0.3)
    assert np.isclose(Q_test[1, 0], 0.8), f"❌ Después de +0.3, esperado: 0.8, obtenido: {Q_test[1, 0]}"
    
    # Test 3: Otros valores no cambian
    assert Q_test[0, 0] == 0.0, "❌ Otros valores no deben cambiar"
    assert Q_test[2, 1] == 0.0, "❌ Otros valores no deben cambiar"
    
    print("✅ Test pasado! Actualizas Q-values correctamente.")
    return True


def accion_aleatoria_test(accion_aleatoria):
    """Test para ejercicio 5: accion_aleatoria"""
    print("Testing accion_aleatoria...")
    
    # Test 1: Rango correcto
    acciones = [accion_aleatoria(4) for _ in range(100)]
    assert all(0 <= a < 4 for a in acciones), "❌ Todas las acciones deben estar entre 0 y 3"
    
    # Test 2: Variedad (al menos 2 acciones diferentes en 100 intentos)
    acciones_unicas = len(set(acciones))
    assert acciones_unicas >= 2, "❌ Debe generar acciones variadas (aleatorias)"
    
    # Test 3: Tipo correcto
    accion = accion_aleatoria(5)
    assert isinstance(accion, (int, np.integer)), f"❌ Debe retornar un entero, obtenido: {type(accion)}"
    
    print("✅ Test pasado! Generas acciones aleatorias correctamente.")
    return True


def promedio_recompensas_test(promedio_recompensas):
    """Test para ejercicio 6: promedio_recompensas"""
    print("Testing promedio_recompensas...")
    
    # Test 1: Promedio simple
    recompensas1 = [10, 20, 15, 30, 25]
    prom1 = promedio_recompensas(recompensas1)
    assert prom1 == 20.0, f"❌ Esperado: 20.0, obtenido: {prom1}"
    
    # Test 2: Otro promedio
    recompensas2 = [5, 5, 5, 5]
    prom2 = promedio_recompensas(recompensas2)
    assert prom2 == 5.0, f"❌ Esperado: 5.0, obtenido: {prom2}"
    
    # Test 3: Con decimales
    recompensas3 = [1, 2, 3]
    prom3 = promedio_recompensas(recompensas3)
    assert prom3 == 2.0, f"❌ Esperado: 2.0, obtenido: {prom3}"
    
    print("✅ Test pasado! Calculas promedios correctamente.")
    return True


if __name__ == "__main__":
    print("Ejecutando todos los tests...")
    print("=" * 50)
    
    # Aquí podrías ejecutar todos los tests si tuvieras las funciones
    print("\nPara ejecutar los tests, importa las funciones desde el notebook")
    print("y llama a cada función de test.")
