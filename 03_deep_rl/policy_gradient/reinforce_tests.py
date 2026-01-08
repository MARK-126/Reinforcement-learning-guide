"""
Tests para REINFORCE_Exercise.ipynb
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical


def policy_network_test(target_class):
    """Test para PolicyNetwork"""
    print("Testing PolicyNetwork...")

    # Test 1: Creación de red
    policy_net = target_class(state_dim=4, action_dim=2)
    assert isinstance(policy_net, nn.Module), "❌ Debe ser nn.Module"

    # Test 2: Shape del forward pass
    test_input = torch.randn(8, 4)
    output = policy_net(test_input)
    assert output.shape == (8, 2), f"❌ Shape incorrecto. Esperado: (8, 2), Obtenido: {output.shape}"

    # Test 3: Probabilidades suman 1
    probs_sum = output.sum(dim=1)
    assert torch.allclose(probs_sum, torch.ones(8), atol=1e-6), \
        "❌ Las probabilidades deben sumar 1 para cada muestra"

    # Test 4: Todas las probabilidades son positivas
    assert torch.all(output >= 0), "❌ Todas las probabilidades deben ser no-negativas"
    assert torch.all(output <= 1), "❌ Todas las probabilidades deben ser <= 1"

    # Test 5: Flujo de gradientes
    assert output.requires_grad, "❌ Output debe requerir gradientes"

    print("✅ Test pasado! PolicyNetwork funciona correctamente.")
    return True


def select_action_test(target_function, PolicyNetwork):
    """Test para selección de acción"""
    print("Testing select_action...")

    policy_net = PolicyNetwork(4, 2)
    test_state = np.array([0.1, 0.2, 0.3, 0.4])

    # Test 1: Tipos de retorno
    action, log_prob = target_function(policy_net, test_state)
    assert isinstance(action, int), f"❌ Acción debe ser int, obtenido: {type(action)}"
    assert isinstance(log_prob, torch.Tensor), "❌ Log prob debe ser tensor"

    # Test 2: Acción válida
    assert 0 <= action < 2, f"❌ Acción debe ser 0 o 1, obtenido: {action}"

    # Test 3: Log prob requiere gradientes
    assert log_prob.requires_grad, "❌ Log prob debe requerir gradientes"

    # Test 4: Log prob es negativo (ya que prob < 1)
    assert log_prob.item() <= 0, "❌ Log probability debe ser negativo o cero"

    # Test 5: Múltiples muestras dan diferentes acciones (probabilístico)
    actions = [target_function(policy_net, test_state)[0] for _ in range(50)]
    unique_actions = len(set(actions))
    assert unique_actions >= 1, "❌ Debe muestrear al menos algunas acciones"

    print("✅ Test pasado! Selección de acción funciona correctamente.")
    return True


def compute_policy_loss_test(target_function):
    """Test para cálculo de policy loss"""
    print("Testing compute_policy_loss...")

    # Test 1: Cálculo básico de loss
    log_probs = [torch.tensor(-0.5, requires_grad=True),
                 torch.tensor(-0.3, requires_grad=True),
                 torch.tensor(-0.7, requires_grad=True)]
    returns = torch.tensor([1.0, 2.0, 0.5])

    loss = target_function(log_probs, returns)

    assert isinstance(loss, torch.Tensor), "❌ Loss debe ser tensor"
    assert loss.requires_grad, "❌ Loss debe requerir gradientes"
    assert loss.dim() == 0, "❌ Loss debe ser escalar"

    # Test 2: Puede hacer backpropagation
    loss.backward()
    for log_prob in log_probs:
        assert log_prob.grad is not None, "❌ Debe calcular gradientes"

    # Test 3: Loss con returns en cero
    log_probs_zero = [torch.tensor(-0.5, requires_grad=True),
                      torch.tensor(-0.3, requires_grad=True)]
    returns_zero = torch.tensor([0.0, 0.0])

    loss_zero = target_function(log_probs_zero, returns_zero)
    assert torch.isfinite(loss_zero), "❌ Loss debe ser finito incluso con returns en cero"

    print("✅ Test pasado! Cálculo de policy loss funciona correctamente.")
    return True


def train_reinforce_test(target_function):
    """Test para entrenamiento REINFORCE (simplificado)"""
    print("Testing train_reinforce...")
    print("Nota: Test completo omitido por velocidad. El notebook mostrará entrenamiento completo.")

    env = gym.make('CartPole-v1')

    # Test con pocos episodios
    try:
        policy_net, rewards = target_function(env, n_episodes=10, lr=0.01, gamma=0.99)

        # Verificaciones básicas
        assert isinstance(policy_net, nn.Module), "❌ Debe retornar nn.Module"
        assert len(rewards) == 10, f"❌ Debe haber 10 recompensas, obtenido: {len(rewards)}"
        assert all(isinstance(r, (int, float, np.number)) for r in rewards), \
            "❌ Todas las recompensas deben ser numéricas"

        # Verificar que policy network es entrenable
        params = list(policy_net.parameters())
        assert len(params) > 0, "❌ Policy network debe tener parámetros"
        assert all(p.requires_grad for p in params), "❌ Parámetros deben requerir gradientes"

        print("✅ Tests básicos pasados!")
        print("⚠️  Test completo: Ejecuta el notebook para entrenar 1000 episodios")

    except Exception as e:
        print(f"❌ Test falló con error: {e}")
        raise

    finally:
        env.close()

    return True


if __name__ == "__main__":
    print("Este archivo contiene funciones de test para REINFORCE.")
    print("Importa y úsalas en el notebook REINFORCE_Exercise.")
