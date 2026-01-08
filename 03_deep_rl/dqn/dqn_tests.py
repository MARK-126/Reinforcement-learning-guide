"""
Public tests for DQN Exercise

This file contains test functions to validate student implementations
of the Deep Q-Network (DQN) algorithm.
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque, namedtuple

# Define Experience tuple (needed for tests)
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


def qnetwork_test(target_class):
    """Test para para para Q-Network"""
    print("Testing QNetwork...")

    # Test 1: Network creation
    q_net = target_class(state_dim=4, action_dim=2, hidden_dim=64)
    assert isinstance(q_net, nn.Module), "❌ QNetwork debe ser a PyTorch nn.Module"

    # Test 2: Forward pass shape
    test_input = torch.randn(8, 4)  # batch_size=8, state_dim=4
    output = q_net(test_input)

    assert output.shape == (8, 2), \
        f"Output shape wrong. Esperado (8, 2), obtenido {output.shape}"

    # Test 3: Single state
    single_state = torch.randn(1, 4)
    output_single = q_net(single_state)
    assert output_single.shape == (1, 2), \
        f"Single state output shape wrong. Esperado (1, 2), obtenido {output_single.shape}"

    # Test 4: Gradient flow
    assert output.requires_grad, "Output debe require gradients"

    # Test 5: Different hidden dimensions
    q_net_large = target_class(state_dim=10, action_dim=5, hidden_dim=256)
    test_input_large = torch.randn(4, 10)
    output_large = q_net_large(test_input_large)
    assert output_large.shape == (4, 5), \
        f"Large network output shape wrong. Esperado (4, 5), obtenido {output_large.shape}"

    print("✅ Test pasado!")


def replay_buffer_test(target_class):
    """Test para para para ReplayBuffer"""
    print("Testing ReplayBuffer...")

    # Test 1: Buffer creation and capacity
    buffer = target_class(capacity=100)
    assert len(buffer) == 0, "Buffer debe start empty"

    # Test 2: Adding experiences
    for i in range(10):
        state = np.array([i, i+1, i+2, i+3])
        action = i % 2
        reward = float(i)
        next_state = state + 1
        done = False
        buffer.add(state, action, reward, next_state, done)

    assert len(buffer) == 10, f"Buffer length wrong. Esperado 10, obtenido {len(buffer)}"

    # Test 3: Sampling batch
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=5)

    assert states.shape == (5, 4), \
        f"States shape wrong. Esperado (5, 4), obtenido {states.shape}"
    assert actions.shape == (5,), \
        f"Actions shape wrong. Esperado (5,), obtenido {actions.shape}"
    assert rewards.shape == (5,), \
        f"Rewards shape wrong. Esperado (5,), obtenido {rewards.shape}"
    assert next_states.shape == (5, 4), \
        f"Next states shape wrong. Esperado (5, 4), obtenido {next_states.shape}"
    assert dones.shape == (5,), \
        f"Dones shape wrong. Esperado (5,), obtenido {dones.shape}"

    # Test 4: Check data types
    assert isinstance(states, np.ndarray), "❌ States debe ser numpy array"
    assert isinstance(actions, np.ndarray), "❌ Actions debe ser numpy array"
    assert isinstance(rewards, np.ndarray), "❌ Rewards debe ser numpy array"
    assert isinstance(next_states, np.ndarray), "❌ Next states debe ser numpy array"
    assert isinstance(dones, np.ndarray), "❌ Dones debe ser numpy array"

    # Test 5: Capacity limit
    buffer_small = target_class(capacity=5)
    for i in range(10):
        state = np.array([i])
        buffer_small.add(state, 0, 0.0, state, False)

    assert len(buffer_small) == 5, \
        f"Buffer debe not exceed capacity. Esperado 5, obtenido {len(buffer_small)}"

    # Test 6: Randomness in sampling
    # Sample multiple times and check we get different samples
    buffer_large = target_class(capacity=100)
    for i in range(50):
        state = np.array([i, i+1])
        buffer_large.add(state, i % 2, float(i), state + 1, False)

    sample1 = buffer_large.sample(10)
    sample2 = buffer_large.sample(10)

    # At least one element debe ser different (with high probability)
    different = not np.allclose(sample1[0], sample2[0])
    assert different, "❌ Samples debe ser random"

    print("✅ Test pasado!")


def compute_td_loss_test(target_function, QNetwork):
    """Test para para TD loss computation"""
    print("Testing compute_td_loss...")

    # Create simple networks
    q_net = QNetwork(state_dim=4, action_dim=2, hidden_dim=32)
    target_net = QNetwork(state_dim=4, action_dim=2, hidden_dim=32)
    target_net.load_state_dict(q_net.state_dict())

    # Create simple batch
    batch_size = 4
    states = np.random.randn(batch_size, 4)
    actions = np.array([0, 1, 0, 1])
    rewards = np.array([1.0, 0.5, -1.0, 2.0])
    next_states = np.random.randn(batch_size, 4)
    dones = np.array([False, False, True, False])

    batch = (states, actions, rewards, next_states, dones)

    # Test 1: Loss computation
    loss = target_function(q_net, target_net, batch, gamma=0.99)

    assert isinstance(loss, torch.Tensor), "❌ Loss debe ser a torch.Tensor"
    assert loss.dim() == 0, "❌ Loss debe ser a scalar (0-dim tensor)"
    assert loss.item() >= 0, "❌ Loss debe ser non-negative"

    # Test 2: Loss has gradient
    assert loss.requires_grad, "Loss debe require gradients for backprop"

    # Test 3: Gradient flow
    q_net.zero_grad()
    loss.backward()

    # Check that at least some parameters have gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in q_net.parameters())
    assert has_grad, "Q-network parameters debe have gradients after backward pass"

    # Test 4: Terminal states (done=True debe ignore next Q-values)
    # Todas terminal states
    dones_terminal = np.array([True, True, True, True])
    batch_terminal = (states, actions, rewards, next_states, dones_terminal)

    loss_terminal = target_function(q_net, target_net, batch_terminal, gamma=0.99)
    assert loss_terminal.item() >= 0, "❌ Loss for terminal states debe ser non-negative"

    print("✅ Test pasado!")


def train_dqn_test(target_function):
    """Test para para DQN training loop (simplified test)"""
    print("Testing train_dqn...")
    print("Nota: Full training test is skipped for speed. Run the notebook to see full training.")

    import gymnasium as gym

    # Create simple environment
    env = gym.make('CartPole-v1')

    # Test with very few episodes
    try:
        q_network, rewards_history = target_function(
            env,
            n_episodes=5,  # Just a few episodes to test
            batch_size=16,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.5,
            epsilon_decay=0.9,
            lr=0.001,
            target_update=2,
            buffer_size=1000
        )

        # Verificaciones básicas
        assert isinstance(q_network, nn.Module), "Debe return a PyTorch module"
        assert len(rewards_history) == 5, \
            f"Rewards history length wrong. Esperado 5, obtenido {len(rewards_history)}"
        assert all(isinstance(r, (int, float, np.number)) for r in rewards_history), \
            "❌ Todas rewards debe ser numeric"

        print("✅ Tests básicos pasados!")
        print("Full training test: Run the notebook to train for 500 episodes")

    except Exception as e:
        print(f"Test para failed with error: {e}")
        raise

    finally:
        env.close()


if __name__ == "__main__":
    print("Este archivo contiene funciones de test para DQN.")
    print("Importa y úsalas en el notebook DQN Exercise.")
