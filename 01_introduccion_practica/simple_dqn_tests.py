"""
Tests para simple_dqn.ipynb
"""
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym


def crear_red_q_test(crear_red_q):
    """Test para ejercicio 1: crear_red_q"""
    print("Testing crear_red_q...")
    
    # Test 1: Retorna nn.Module
    red = crear_red_q(estado_dim=4, accion_dim=2)
    assert isinstance(red, nn.Module), "❌ Debe retornar nn.Module"
    
    # Test 2: Tiene las capas correctas
    assert hasattr(red, 'fc1'), "❌ Debe tener capa fc1"
    assert hasattr(red, 'fc2'), "❌ Debe tener capa fc2"
    
    # Test 3: Dimensiones correctas
    assert red.fc1.in_features == 4, "❌ fc1 input incorrecto"
    assert red.fc2.out_features == 2, "❌ fc2 output incorrecto"
    
    print("✅ Test pasado! Red Q creada correctamente.")
    return True


def predecir_q_values_test(predecir_q_values, crear_red_q):
    """Test para ejercicio 2: predecir_q_values"""
    print("Testing predecir_q_values...")
    
    red = crear_red_q(4, 2)
    estado = np.array([0.1, 0.2, 0.3, 0.4])
    
    # Test 1: Retorna tensor
    q_values = predecir_q_values(red, estado)
    assert isinstance(q_values, torch.Tensor), "❌ Debe retornar tensor"
    
    # Test 2: Shape correcto
    assert q_values.shape == torch.Size([2]), f"❌ Shape incorrecto: {q_values.shape}"
    
    # Test 3: Valores numéricos
    assert not torch.isnan(q_values).any(), "❌ Q-values contienen NaN"
    
    print("✅ Test pasado! Predicción de Q-values funciona.")
    return True


def calcular_dqn_loss_test(calcular_dqn_loss):
    """Test para ejercicio 3: calcular_dqn_loss"""
    print("Testing calcular_dqn_loss...")
    
    q_pred = torch.tensor([1.0, 2.0])
    q_target = torch.tensor([1.5, 2.5])
    
    # Test 1: Retorna tensor
    loss = calcular_dqn_loss(q_pred, q_target)
    assert isinstance(loss, torch.Tensor), "❌ Debe retornar tensor"
    
    # Test 2: Loss correcto (MSE)
    expected_loss = ((1.0-1.5)**2 + (2.0-2.5)**2) / 2
    assert abs(loss.item() - expected_loss) < 0.001, \
        f"❌ Loss incorrecto. Esperado: {expected_loss}, Obtenido: {loss.item()}"
    
    # Test 3: Loss es escalar
    assert loss.dim() == 0, "❌ Loss debe ser escalar"
    
    print("✅ Test pasado! Cálculo de loss funciona.")
    return True


def entrenar_paso_dqn_test(entrenar_paso_dqn, crear_red_q):
    """Test para ejercicio 4: entrenar_paso_dqn"""
    print("Testing entrenar_paso_dqn...")
    
    red = crear_red_q(4, 2)
    optimizer = torch.optim.Adam(red.parameters(), lr=0.01)
    
    estado = np.array([0.1, 0.2, 0.3, 0.4])
    siguiente = np.array([0.2, 0.3, 0.4, 0.5])
    
    # Guardar pesos iniciales
    initial_weights = red.fc1.weight.data.clone()
    
    # Test 1: Retorna loss
    loss = entrenar_paso_dqn(red, optimizer, estado, accion=0, recompensa=1.0,
                             siguiente_estado=siguiente, done=False)
    assert isinstance(loss, torch.Tensor), "❌ Debe retornar loss"
    
    # Test 2: Pesos cambiaron
    assert not torch.equal(initial_weights, red.fc1.weight.data), \
        "❌ Los pesos deben cambiar después del entrenamiento"
    
    # Test 3: Loss disminuye con múltiples pasos
    losses = []
    for _ in range(10):
        loss = entrenar_paso_dqn(red, optimizer, estado, accion=0, recompensa=1.0,
                                 siguiente_estado=siguiente, done=False)
        losses.append(loss.item())
    
    # Al menos debería haber alguna tendencia a bajar
    assert losses[-1] <= losses[0] * 1.5, \
        "❌ Loss debería tender a disminuir con el entrenamiento"
    
    print("✅ Test pasado! Paso de entrenamiento funciona.")
    return True


def entrenar_dqn_simple_test(entrenar_dqn_simple):
    """Test para ejercicio 5: entrenar_dqn_simple"""
    print("Testing entrenar_dqn_simple...")
    print("Nota: Test simplificado por velocidad. El notebook mostrará entrenamiento completo.")
    
    env = gym.make('CartPole-v1')
    
    # Test con pocos episodios
    try:
        red, recompensas = entrenar_dqn_simple(env, n_episodios=10, epsilon=0.1, lr=0.01)
        
        # Verificaciones básicas
        assert isinstance(red, nn.Module), "❌ Debe retornar nn.Module"
        assert len(recompensas) == 10, f"❌ Debe haber 10 recompensas, obtenido: {len(recompensas)}"
        assert all(isinstance(r, (int, float, np.number)) for r in recompensas), \
            "❌ Todas las recompensas deben ser numéricas"
        
        # Verificar que la red es entrenable
        params = list(red.parameters())
        assert len(params) > 0, "❌ Red debe tener parámetros"
        assert all(p.requires_grad for p in params), "❌ Parámetros deben requerir gradientes"
        
        print("✅ Tests básicos pasados!")
        print("⚠️  Test completo: Ejecuta el notebook para entrenar 200 episodios")
        
    except Exception as e:
        print(f"❌ Test falló con error: {e}")
        raise
    
    finally:
        env.close()
    
    return True


if __name__ == "__main__":
    print("Este archivo contiene funciones de test para simple_dqn.ipynb")
    print("Importa y úsalas en el notebook.")
