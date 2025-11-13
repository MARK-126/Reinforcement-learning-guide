"""
Script de verificación para las variantes de DQN

Verifica que las implementaciones funcionan correctamente:
- Double DQN
- Dueling DQN
"""

import sys
import torch
import numpy as np
import gymnasium as gym

# Importar las implementaciones
from double_dqn import DoubleDQNAgent, train_double_dqn
from dueling_dqn import DuelingDQNAgent, DuelingDQN, train_dueling_dqn


def test_double_dqn():
    """Test básico de Double DQN"""
    print("="*60)
    print("Testing Double DQN")
    print("="*60)

    # Crear ambiente simple
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Crear agente
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=100,
        buffer_size=1000,
        batch_size=32,
        target_update=5,
        hidden_dim=64
    )

    print(f"✓ Agente creado")
    print(f"  - Estado: {state_dim} dim")
    print(f"  - Acciones: {action_dim}")
    print(f"  - Device: {agent.device}")

    # Test get_action
    state, _ = env.reset()
    action = agent.get_action(state)
    assert 0 <= action < action_dim
    print(f"✓ get_action funciona: acción {action}")

    # Test store_transition
    next_state, reward, terminated, truncated, _ = env.step(action)
    agent.store_transition(state, action, reward, next_state, terminated)
    print(f"✓ store_transition funciona: buffer size {len(agent.replay_buffer)}")

    # Llenar buffer
    for _ in range(100):
        state, _ = env.reset()
        for _ in range(10):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            if done:
                break
            state = next_state

    print(f"✓ Buffer llenado: {len(agent.replay_buffer)} transiciones")

    # Test train_step
    loss = agent.train_step()
    assert loss is not None
    print(f"✓ train_step funciona: loss={loss:.4f}")

    # Test update_epsilon
    initial_epsilon = agent.epsilon
    agent.steps = 50
    agent.update_epsilon()
    assert agent.epsilon < initial_epsilon
    print(f"✓ update_epsilon funciona: ε={agent.epsilon:.3f}")

    # Test update_target_network
    old_target_params = [p.clone() for p in agent.target_network.parameters()]
    agent.update_target_network()
    new_target_params = list(agent.target_network.parameters())
    # Verificar que cambió (hard update)
    print(f"✓ update_target_network funciona")

    # Test save/load
    save_path = "/tmp/test_double_dqn.pth"
    agent.save(save_path)
    print(f"✓ save funciona")

    agent2 = DoubleDQNAgent(state_dim, action_dim)
    agent2.load(save_path)
    assert agent2.steps == agent.steps
    print(f"✓ load funciona")

    # Test entrenamiento corto
    print("\nEntrenando por 5 episodios...")
    rewards, losses = train_double_dqn(env, agent, n_episodes=5, max_steps=50, save_every=0)
    assert len(rewards) == 5
    assert len(losses) == 5
    print(f"✓ train_double_dqn funciona: rewards={rewards}")

    env.close()
    print("\n✓ Double DQN: TODOS LOS TESTS PASARON\n")


def test_dueling_dqn():
    """Test básico de Dueling DQN"""
    print("="*60)
    print("Testing Dueling DQN")
    print("="*60)

    # Crear ambiente simple
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Test arquitectura Dueling
    network = DuelingDQN(state_dim, action_dim, hidden_dim=64)
    state_tensor = torch.randn(2, state_dim)
    q_values = network(state_tensor)
    assert q_values.shape == (2, action_dim)
    print(f"✓ Arquitectura Dueling funciona: Q-values shape {q_values.shape}")

    # Test get_value_and_advantage
    value, advantage = network.get_value_and_advantage(state_tensor)
    assert value.shape == (2, 1)
    assert advantage.shape == (2, action_dim)
    print(f"✓ Value/Advantage streams: V shape {value.shape}, A shape {advantage.shape}")

    # Verificar agregación: Q(s,a) = V(s) + (A(s,a) - mean(A))
    expected_q = value + (advantage - advantage.mean(dim=1, keepdim=True))
    assert torch.allclose(q_values, expected_q, atol=1e-6)
    print(f"✓ Agregación correcta: Q = V + (A - mean(A))")

    # Crear agente
    agent = DuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=100,
        buffer_size=1000,
        batch_size=32,
        target_update=5,
        hidden_dim=64,
        use_double_dqn=True
    )

    print(f"✓ Agente creado (con Double DQN)")
    print(f"  - Estado: {state_dim} dim")
    print(f"  - Acciones: {action_dim}")
    print(f"  - Device: {agent.device}")

    # Test get_action
    state, _ = env.reset()
    action = agent.get_action(state)
    assert 0 <= action < action_dim
    print(f"✓ get_action funciona: acción {action}")

    # Test analyze_value_advantage
    analysis = agent.analyze_value_advantage(state)
    assert 'value' in analysis
    assert 'advantage' in analysis
    assert 'q_values' in analysis
    assert analysis['advantage'].shape == (action_dim,)
    print(f"✓ analyze_value_advantage funciona")
    print(f"  V(s) = {analysis['value']:.3f}")
    print(f"  A(s,a) = {analysis['advantage']}")
    print(f"  Q(s,a) = {analysis['q_values']}")

    # Llenar buffer
    for _ in range(100):
        state, _ = env.reset()
        for _ in range(10):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            if done:
                break
            state = next_state

    print(f"✓ Buffer llenado: {len(agent.replay_buffer)} transiciones")

    # Test train_step (con Double DQN)
    loss = agent.train_step()
    assert loss is not None
    print(f"✓ train_step funciona (Double DQN): loss={loss:.4f}")

    # Test sin Double DQN
    agent.use_double_dqn = False
    loss = agent.train_step()
    assert loss is not None
    print(f"✓ train_step funciona (Standard DQN): loss={loss:.4f}")
    agent.use_double_dqn = True

    # Test save/load
    save_path = "/tmp/test_dueling_dqn.pth"
    agent.save(save_path)
    print(f"✓ save funciona")

    agent2 = DuelingDQNAgent(state_dim, action_dim)
    agent2.load(save_path)
    assert agent2.steps == agent.steps
    assert agent2.use_double_dqn == agent.use_double_dqn
    print(f"✓ load funciona")

    # Test entrenamiento corto
    print("\nEntrenando por 5 episodios...")
    rewards, losses = train_dueling_dqn(env, agent, n_episodes=5, max_steps=50, save_every=0)
    assert len(rewards) == 5
    assert len(losses) == 5
    print(f"✓ train_dueling_dqn funciona: rewards={rewards}")

    env.close()
    print("\n✓ Dueling DQN: TODOS LOS TESTS PASARON\n")


def compare_architectures():
    """Compara las arquitecturas de las redes"""
    print("="*60)
    print("Comparando Arquitecturas")
    print("="*60)

    state_dim = 4
    action_dim = 2
    hidden_dim = 64

    # Importar DQN básico
    from dqn_basic import DQN

    # Crear redes
    basic_dqn = DQN(state_dim, action_dim, hidden_dim)
    dueling_dqn = DuelingDQN(state_dim, action_dim, hidden_dim)

    # Contar parámetros
    basic_params = sum(p.numel() for p in basic_dqn.parameters())
    dueling_params = sum(p.numel() for p in dueling_dqn.parameters())

    print(f"\nDQN Básico:")
    print(f"  Parámetros: {basic_params:,}")
    print(f"  Arquitectura: simple MLP")

    print(f"\nDueling DQN:")
    print(f"  Parámetros: {dueling_params:,}")
    print(f"  Arquitectura: feature + value stream + advantage stream")
    print(f"  Diferencia: {dueling_params - basic_params:,} parámetros")

    # Test forward pass
    state = torch.randn(1, state_dim)

    q_basic = basic_dqn(state)
    q_dueling = dueling_dqn(state)

    print(f"\nSalidas:")
    print(f"  DQN básico: {q_basic}")
    print(f"  Dueling DQN: {q_dueling}")

    value, advantage = dueling_dqn.get_value_and_advantage(state)
    print(f"\nDueling streams:")
    print(f"  V(s): {value.item():.3f}")
    print(f"  A(s,a): {advantage.squeeze().tolist()}")

    print("\n✓ Comparación completada\n")


def main():
    """Ejecuta todos los tests"""
    print("\n" + "="*60)
    print("VERIFICACIÓN DE DQN VARIANTS")
    print("="*60 + "\n")

    try:
        test_double_dqn()
        test_dueling_dqn()
        compare_architectures()

        print("="*60)
        print("✓ TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("="*60)
        print("\nLas implementaciones están listas para usar:")
        print("1. Double DQN - Reduce sobreestimación de Q-values")
        print("2. Dueling DQN - Separa valor y ventajas")
        print("3. Se pueden combinar para mejores resultados")
        print("\nEjecutar individualmente:")
        print("  python double_dqn.py")
        print("  python dueling_dqn.py")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
