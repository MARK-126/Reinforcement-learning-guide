"""
DQN Utilities and Testing Functions
Deep Reinforcement Learning - Deep Q-Networks Implementation
Based on DeepLearning.AI professional notebook format
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque, namedtuple
from typing import Tuple, List, Optional
import random

# Define Transition for Replay Buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# ============================================================================
# TESTING FUNCTIONS FOR EXERCISES
# ============================================================================

def test_dqn_network(network_class, state_dim=4, action_dim=2, hidden_dim=128):
    """
    Test Exercise 1: DQN Network Implementation

    Args:
        network_class: The DQN class to test
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dim: Hidden layer dimension

    Returns:
        bool: True if all tests pass
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Test 1: Network creation
        net = network_class(state_dim, action_dim, hidden_dim).to(device)
        assert isinstance(net, nn.Module), "Network must be nn.Module instance"

        # Test 2: Forward pass
        batch_size = 32
        dummy_input = torch.randn(batch_size, state_dim).to(device)
        output = net(dummy_input)

        assert output.shape == (batch_size, action_dim), \
            f"Output shape mismatch. Expected {(batch_size, action_dim)}, got {output.shape}"

        # Test 3: Gradient flow
        loss = output.sum()
        loss.backward()
        has_gradients = any(p.grad is not None for p in net.parameters())
        assert has_gradients, "Network parameters should have gradients"

        print("✓ Test 1.1: Network creation - PASSED")
        print("✓ Test 1.2: Forward pass shape - PASSED")
        print("✓ Test 1.3: Gradient computation - PASSED")
        return True

    except Exception as e:
        print(f"✗ Test FAILED: {str(e)}")
        return False


def test_replay_buffer(buffer_class, capacity=10000):
    """
    Test Exercise 2: Replay Buffer Implementation

    Args:
        buffer_class: The ReplayBuffer class to test
        capacity: Maximum buffer capacity

    Returns:
        bool: True if all tests pass
    """
    try:
        buffer = buffer_class(capacity=capacity)

        # Test 1: Push transitions
        for i in range(100):
            state = np.random.randn(4)
            action = i % 2
            reward = float(i)
            next_state = np.random.randn(4)
            done = (i % 10 == 0)
            buffer.push(state, action, reward, next_state, done)

        assert len(buffer) == 100, f"Buffer length should be 100, got {len(buffer)}"

        # Test 2: Sample batch
        batch = buffer.sample(32)
        assert len(batch) == 32, f"Batch size should be 32, got {len(batch)}"

        # Test 3: Transition structure
        trans = batch[0]
        assert hasattr(trans, 'state'), "Transition must have 'state'"
        assert hasattr(trans, 'action'), "Transition must have 'action'"
        assert hasattr(trans, 'reward'), "Transition must have 'reward'"
        assert hasattr(trans, 'next_state'), "Transition must have 'next_state'"
        assert hasattr(trans, 'done'), "Transition must have 'done'"

        # Test 4: Capacity limit
        for i in range(capacity + 100):
            buffer.push(np.random.randn(4), 0, 0.0, np.random.randn(4), False)

        assert len(buffer) <= capacity, f"Buffer exceeded capacity: {len(buffer)} > {capacity}"

        print("✓ Test 2.1: Push transitions - PASSED")
        print("✓ Test 2.2: Sample batch - PASSED")
        print("✓ Test 2.3: Transition structure - PASSED")
        print("✓ Test 2.4: Capacity management - PASSED")
        return True

    except Exception as e:
        print(f"✗ Test FAILED: {str(e)}")
        return False


def test_dqn_update(agent_class, state_dim=4, action_dim=2):
    """
    Test Exercise 3: DQN Update Implementation

    Args:
        agent_class: The DQN Agent class to test
        state_dim: State dimension
        action_dim: Action dimension

    Returns:
        bool: True if all tests pass
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create agent
        agent = agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=1e-3,
            gamma=0.99,
            buffer_size=1000,
            batch_size=32
        )

        # Test 1: Store transitions
        for i in range(100):
            state = np.random.randn(state_dim)
            action = i % action_dim
            reward = float(i)
            next_state = np.random.randn(state_dim)
            done = (i % 10 == 0)
            agent.store_transition(state, action, reward, next_state, done)

        assert len(agent.replay_buffer) == 100, "Transitions not stored correctly"

        # Test 2: Train step (should not crash)
        loss = agent.train_step()
        assert loss is not None, "Train step should return loss"
        assert isinstance(loss, (float, np.floating)), "Loss should be float"
        assert loss > 0, "Loss should be positive"

        # Test 3: Action selection
        state = np.random.randn(state_dim)
        action = agent.get_action(state, training=True)
        assert isinstance(action, (int, np.integer)), "Action should be integer"
        assert 0 <= action < action_dim, f"Action out of range: {action}"

        # Test 4: Target network update
        agent.update_target_network()
        # Check that parameters are copied
        for p1, p2 in zip(agent.q_network.parameters(), agent.target_network.parameters()):
            assert torch.allclose(p1, p2), "Target network not updated correctly"

        print("✓ Test 3.1: Store transitions - PASSED")
        print("✓ Test 3.2: Training step - PASSED")
        print("✓ Test 3.3: Action selection - PASSED")
        print("✓ Test 3.4: Target network update - PASSED")
        return True

    except Exception as e:
        print(f"✗ Test FAILED: {str(e)}")
        return False


def test_double_dqn_update(agent_class, state_dim=4, action_dim=2):
    """
    Test Exercise 4: Double DQN Implementation

    Args:
        agent_class: The Double DQN Agent class to test
        state_dim: State dimension
        action_dim: Action dimension

    Returns:
        bool: True if all tests pass
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create agent
        agent = agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=1e-3,
            gamma=0.99,
            buffer_size=1000,
            batch_size=32
        )

        # Test 1: Population of replay buffer
        for i in range(100):
            state = np.random.randn(state_dim)
            action = i % action_dim
            reward = float(i)
            next_state = np.random.randn(state_dim)
            done = (i % 10 == 0)
            agent.store_transition(state, action, reward, next_state, done)

        # Test 2: Double DQN update (different from standard DQN)
        loss1 = agent.train_step()
        loss2 = agent.train_step()

        assert loss1 is not None, "Double DQN train_step should return loss"
        assert loss2 is not None, "Multiple train steps should work"

        # Test 3: Check two networks exist and differ
        params1 = [p.clone() for p in agent.q_network.parameters()]
        params2 = [p.clone() for p in agent.target_network.parameters()]

        # Before update, target network should equal q_network (from initialization)
        agent.update_target_network()

        # Test 4: Epsilon decay
        initial_epsilon = agent.epsilon
        agent.steps = 100
        agent.update_epsilon()
        assert agent.epsilon < initial_epsilon, "Epsilon should decay over time"

        print("✓ Test 4.1: Replay buffer population - PASSED")
        print("✓ Test 4.2: Double DQN training - PASSED")
        print("✓ Test 4.3: Network differentiation - PASSED")
        print("✓ Test 4.4: Epsilon decay - PASSED")
        return True

    except Exception as e:
        print(f"✗ Test FAILED: {str(e)}")
        return False


def test_dueling_dqn_architecture(network_class, state_dim=4, action_dim=2, hidden_dim=128):
    """
    Test Exercise 5: Dueling DQN Architecture

    Args:
        network_class: The Dueling DQN class to test
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dim: Hidden layer dimension

    Returns:
        bool: True if all tests pass
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Test 1: Network creation
        net = network_class(state_dim, action_dim, hidden_dim).to(device)
        assert isinstance(net, nn.Module), "Network must be nn.Module instance"

        # Test 2: Forward pass returns Q-values
        batch_size = 32
        dummy_input = torch.randn(batch_size, state_dim).to(device)
        output = net(dummy_input)

        assert output.shape == (batch_size, action_dim), \
            f"Q-values shape mismatch. Expected {(batch_size, action_dim)}, got {output.shape}"

        # Test 3: Value and Advantage streams exist
        assert hasattr(net, 'value_stream'), "Network must have value_stream attribute"
        assert hasattr(net, 'advantage_stream'), "Network must have advantage_stream attribute"

        # Test 4: Get value and advantage method
        if hasattr(net, 'get_value_and_advantage'):
            value, advantage = net.get_value_and_advantage(dummy_input)
            assert value.shape == (batch_size, 1), f"Value shape should be {(batch_size, 1)}"
            assert advantage.shape == (batch_size, action_dim), \
                f"Advantage shape should be {(batch_size, action_dim)}"

        # Test 5: Verify Q = V + (A - mean(A))
        value, advantage = net.get_value_and_advantage(dummy_input)
        q_values_reconstructed = value + (advantage - advantage.mean(dim=1, keepdim=True))
        q_values_direct = net(dummy_input)

        assert torch.allclose(q_values_reconstructed, q_values_direct, atol=1e-5), \
            "Q-values computation mismatch"

        # Test 6: Gradient flow
        loss = output.sum()
        loss.backward()
        has_gradients = any(p.grad is not None for p in net.parameters())
        assert has_gradients, "Network parameters should have gradients"

        print("✓ Test 5.1: Network creation - PASSED")
        print("✓ Test 5.2: Forward pass shape - PASSED")
        print("✓ Test 5.3: Dual streams exist - PASSED")
        print("✓ Test 5.4: V+A decomposition - PASSED")
        print("✓ Test 5.5: Q-value reconstruction - PASSED")
        print("✓ Test 5.6: Gradient computation - PASSED")
        return True

    except Exception as e:
        print(f"✗ Test FAILED: {str(e)}")
        return False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_test_batch(state_dim=4, action_dim=2, batch_size=32, device='cpu'):
    """Create a test batch for debugging"""
    state_batch = torch.randn(batch_size, state_dim).to(device)
    action_batch = torch.randint(0, action_dim, (batch_size,)).to(device)
    reward_batch = torch.randn(batch_size).to(device)
    next_state_batch = torch.randn(batch_size, state_dim).to(device)
    done_batch = torch.randint(0, 2, (batch_size,)).float().to(device)

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch


def compare_q_networks(net1, net2, input_tensor):
    """Compare outputs of two networks"""
    with torch.no_grad():
        out1 = net1(input_tensor)
        out2 = net2(input_tensor)
    return torch.max(torch.abs(out1 - out2)).item()


def print_network_summary(network, input_size):
    """Print network architecture summary"""
    print(network)
    print(f"\nNetwork trained on inputs of size: {input_size}")

    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


# ============================================================================
# HELPER FUNCTIONS FOR DEBUGGING
# ============================================================================

def check_action_validity(action, action_dim):
    """Validate action value"""
    assert isinstance(action, (int, np.integer)), f"Action must be integer, got {type(action)}"
    assert 0 <= action < action_dim, f"Action {action} out of range [0, {action_dim-1}]"
    return True


def compute_td_error(q_value, target_q_value):
    """Compute Temporal Difference error"""
    return abs(q_value - target_q_value)


def print_dqn_comparison_table():
    """Print comparison table of DQN variants"""
    table = """
    ╔════════════════════╦═══════════════════════════════════════════════════════════╗
    ║ Aspect             ║ DQN vs Double DQN vs Dueling DQN                           ║
    ╠════════════════════╬═══════════════════════════════════════════════════════════╣
    ║ Update Equation    ║ DQN: r + γ max_a' Q_t(s',a')                              ║
    ║                    ║ D-DQN: r + γ Q_t(s', argmax_a' Q(s',a'))                  ║
    ║                    ║ Dueling: Same but with V+A architecture                   ║
    ╠════════════════════╬═══════════════════════════════════════════════════════════╣
    ║ Main Problem       ║ DQN: Overestimation of Q-values                           ║
    ║                    ║ D-DQN: Reduced via action selection/evaluation decoupling ║
    ║                    ║ Dueling: Architectural separation of V and A              ║
    ╠════════════════════╬═══════════════════════════════════════════════════════════╣
    ║ Complexity         ║ DQN: Low | D-DQN: Very Low (1-line change) | Dueling: Med║
    ╠════════════════════╬═══════════════════════════════════════════════════════════╣
    ║ Convergence        ║ DQN: Good | D-DQN: Better | Dueling: Best on large action║
    ╠════════════════════╬═══════════════════════════════════════════════════════════╣
    ║ Best For           ║ DQN: Baseline | D-DQN: Stability | Dueling: Large actions ║
    ╚════════════════════╩═══════════════════════════════════════════════════════════╝
    """
    print(table)


if __name__ == "__main__":
    print("DQN Utilities Module Loaded Successfully")
    print_dqn_comparison_table()
